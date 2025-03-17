#!/bin/bash
#SBATCH --job-name=hist_time
#SBATCH --output=hist_time.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000MB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodelist=wn[215]


###############################################################################################################
prg_directory="hist/"
gpu_program="${prg_directory}hist_cuda.cu" # GPU program source file
cpu_program="${prg_directory}hist_cpu.c" # CPU program source file 
image_dir="./resources/input" # set the images directory
num_runs=20  # number of times to test each image
output_file="hist_times.txt" # output file to store the results
###############################################################################################################


echo "Starting tests..."

echo "Creating output file..."
> $output_file  # clear output file
echo "Output file created: $output_file"

echo "Loading CUDA module..."
module load CUDA
echo "CUDA module loaded"

echo "Compiling GPU program..."
srun --partition=gpu nvcc -Iinclude/ $gpu_program -o hist_cuda.out -diag-suppress 55
echo "GPU program compiled"

echo "Compiling CPU program..."
srun gcc -Iinclude/ $cpu_program -o hist_cpu.out -lm
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
    cpu_times=()

    for run in $(seq 1 $num_runs); do
        echo "Run #$run for $image"
        
        # Run the GPU program and capture the printed output
        gpu_output=$(srun --partition=gpu --ntasks=1 --gpus=1 --mem-per-cpu=16000MB hist_cuda.out "$image")
        
        # Run the CPU program (seq) and capture the printed output
        cpu_output=$(srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=16000MB hist_cpu.out "$image")

        # Extract execution times from output
        gpu_kernel_time=$(echo "$gpu_output" | grep "Kernel execution time:" | awk '{print $4}')
        gpu_total_time=$(echo "$gpu_output" | grep "Total execution time:" | awk '{print $4}')
        cpu_time=$(echo "$cpu_output" | grep "CPU execution time:" | awk '{print $4}')

        gpu_kernel_times+=($gpu_kernel_time)
        gpu_total_times+=($gpu_total_time)
        cpu_times+=($cpu_time)

        # Store the result in the output file
        echo "$image, Run #$run: GPU (kernel): $gpu_kernel_time ms, GPU (total): $gpu_total_time ms, CPU: $cpu_time ms" >> $output_file
    done
    
    # Report
    # Calculate averages of ALL runs
    gpu_kernel_avg=$(echo "${gpu_kernel_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    gpu_total_avg=$(echo "${gpu_total_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    cpu_avg=$(echo "${cpu_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')

    # Calculate speedup based on averages of all runs
    if (( $(echo "$gpu_total_avg > 0" | bc -l) )); then
        cpu_vs_gpu=$(echo "scale=4; $cpu_avg / $gpu_total_avg" | bc -l)
    else
        cpu_vs_gpu="N/A"
    fi

    # Extract the full image dimensions line from GPU output
    image_info=$(echo "$gpu_output" | grep "Image size:")

    # Write the summary results to the output file
    echo "-------------------------------- SUMMARY FOR $image --------------------------------" >> $output_file
    echo $image_info >> $output_file
    echo "AVERAGE TIMES (ALL RUNS):" >> $output_file
    echo "GPU kernel: $gpu_kernel_avg ms, GPU total: $gpu_total_avg ms" >> $output_file
    echo "CPU: $cpu_avg ms" >> $output_file
    echo "SPEEDUPS:" >> $output_file
    echo "CPU vs GPU total: $cpu_vs_gpu x faster (including data transfers)" >> $output_file
    echo "--------------------------------------------------------------------------------" >> $output_file
    echo "" >> $output_file # add an empty line after each image's results
    echo "Tests for $image completed"

done

echo "All tests completed"