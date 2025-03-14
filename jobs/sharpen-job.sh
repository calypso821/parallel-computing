#!/bin/bash
#SBATCH --job-name=sharpen_time
#SBATCH --output=sharpen_time.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000MB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1


###############################################################################################################
prg_directory="image_proc/"
gpu_program="${prg_directory}sharpen_filter.cu" # GPU program source file
cpu_seq_program="${prg_directory}sharpen_filter_seq.c" # CPU program (sequential) source file 
cpu_par_program="${prg_directory}sharpen_filter_par.c" # CPU program (parallel) source file 
image_dir="./resources/input" # set the images directory
num_runs=20  # number of times to test each image
output_file="sharpen_times.txt" # output file to store the results
###############################################################################################################


echo "Starting tests..."

echo "Creating output file..."
> $output_file  # clear output file
echo "Output file created: $output_file"

echo "Loading CUDA module..."
module load CUDA
echo "CUDA module loaded"

echo "Compiling GPU program..."
srun --partition=gpu nvcc $gpu_program -o sharpen_cuda.out -diag-suppress 55
echo "GPU program compiled"

echo "Compiling CPU program..."
srun gcc $cpu_seq_program -o sharpen_cpu_seq.out -lm
srun gcc $cpu_par_program -o sharpen_cpu_par.out -lm
echo "CPU program compiled"

# Loop through all images in the image dir
for image in "$image_dir"/*.{jpg,jpeg,png,bmp}; do
    if [ ! -e "$image" ]; then
        echo "Directory processed."
        break
    fi
    echo "Running tests for $image"
    
    # Empty arrays to store times
    gpu_kernel_times=()
    gpu_total_times=()
    cpu_seq_times=()
    cpu_par_times=()

    for run in $(seq 1 $num_runs); do
        echo "Run #$run for $image"
        
        # Run the GPU program and capture the printed output
        gpu_output=$(srun --partition=gpu --ntasks=1 --gpus=1 --mem-per-cpu=16000MB sharpen_cuda.out "$image")
        
        # Run the CPU program (seq) and capture the printed output
        cpu_seq_output=$(srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=16000MB sharpen_cpu_seq.out "$image")
    
        # Run the CPU program (par) and capture the printed output
        cpu_par_output=$(srun --ntasks=1 --cpus-per-task=4 --mem-per-cpu=16000MB sharpen_cpu_par.out "$image")  
        
        # Extract execution times from output
        gpu_kernel_time=$(echo "$gpu_output" | grep "Kernel execution time:" | awk '{print $4}')
        gpu_total_time=$(echo "$gpu_output" | grep "Total execution time:" | awk '{print $4}')
        cpu_seq_time=$(echo "$cpu_seq_output" | grep "CPU (seq) execution time:" | awk '{print $5}')
        cpu_par_time=$(echo "$cpu_par_output" | grep "CPU (par) execution time:" | awk '{print $5}')

        gpu_kernel_times+=($gpu_kernel_time)
        gpu_total_times+=($gpu_total_time)
        cpu_par_times+=($cpu_par_time)
        cpu_seq_times+=($cpu_seq_time)

        # Store the result in the output file
        echo "$image, Run #$run: GPU (kernel): $gpu_kernel_time ms, GPU (total): $gpu_total_time ms, CPU (seq): $cpu_seq_time ms, CPU (par): $cpu_par_time ms" >> $output_file
    done

    # Report
    # Sort times in ascending order
    sorted_gpu_kernel=($(printf '%s\n' "${gpu_kernel_times[@]}" | sort -n))
    sorted_gpu_total=($(printf '%s\n' "${gpu_total_times[@]}" | sort -n))
    sorted_cpu_seq=($(printf '%s\n' "${cpu_seq_times[@]}" | sort -n))
    sorted_cpu_par=($(printf '%s\n' "${cpu_par_times[@]}" | sort -n))

    # Calculate how many values represent 20% (at least 1)
    num_runs=${#gpu_kernel_times[@]}
    best_count=$(echo "($num_runs * 0.2 + 0.5) / 1" | bc)
    best_count=$((best_count > 0 ? best_count : 1))

    # Take the best 20% and calculate their average
    best_gpu_kernel=("${sorted_gpu_kernel[@]:0:$best_count}")
    best_gpu_total=("${sorted_gpu_total[@]:0:$best_count}")
    best_cpu_seq=("${sorted_cpu_seq[@]:0:$best_count}")
    best_cpu_par=("${sorted_cpu_par[@]:0:$best_count}")

    # Calculate averages of the best 20%
    gpu_kernel_best=$(echo "${best_gpu_kernel[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    gpu_total_best=$(echo "${best_gpu_total[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    cpu_seq_best=$(echo "${best_cpu_seq[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    cpu_par_best=$(echo "${best_cpu_par[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')

    # Calculate speedups based on best 20% averages
    # Using GPU total time for CPU vs GPU comparisons
    if (( $(echo "$gpu_total_best > 0" | bc -l) )); then
        seq_vs_gpu=$(echo "scale=4; $cpu_seq_best / $gpu_total_best" | bc -l)
        par_vs_gpu=$(echo "scale=4; $cpu_par_best / $gpu_total_best" | bc -l)
    else
        seq_vs_gpu="N/A"
        par_vs_gpu="N/A"
    fi

    # Calculate seq vs par speedup
    if (( $(echo "$cpu_par_best > 0" | bc -l) )); then
        seq_vs_par=$(echo "scale=4; $cpu_seq_best / $cpu_par_best" | bc -l)
    else
        seq_vs_par="N/A"
    fi

    # Write the summary results to the output file
    echo "-------------------------------- SUMMARY FOR $image --------------------------------" >> $output_file
    echo "AVERAGE TIMES (BEST 20%):" >> $output_file
    echo "GPU kernel: $gpu_kernel_best ms, GPU total: $gpu_total_best ms" >> $output_file
    echo "CPU seq: $cpu_seq_best ms, CPU par: $cpu_par_best ms" >> $output_file
    echo "SPEEDUPS:" >> $output_file
    echo "CPU seq vs GPU total: $seq_vs_gpu x (including data transfers)" >> $output_file
    echo "CPU par vs GPU total: $par_vs_gpu x (including data transfers)" >> $output_file
    echo "CPU seq vs CPU par: $seq_vs_par x" >> $output_file
    echo "--------------------------------------------------------------------------------" >> $output_file
    echo "" >> $output_file  # add an empty line after each image's results
    echo "Tests for $image completed"


done

echo "All tests completed"