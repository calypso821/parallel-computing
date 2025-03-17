#!/bin/bash
#SBATCH --job-name=hist_kernel_time
#SBATCH --output=hist_kernel_time.out
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
image_dir="./resources/input" # set the images directory
num_runs=20  # number of times to test each image
output_file="hist_kernel_times.txt" # output file to store the results
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

# Loop through all images in the image dir
for image in "$image_dir"/*.{jpg,jpeg,png,bmp}; do
    if [ ! -e "$image" ]; then
        echo "Directory processed."
        break
    fi
    echo "Running tests for $image"
    
    # Empty arrays to store times
    hist_naive_times=()
    hist_local_times=()
    cdf_naive_times=()
    cdf_blelloch_times=()

    for run in $(seq 1 $num_runs); do
        echo "Run #$run for $image"
        
        # Run the GPU program and capture the printed output
        gpu_output=$(srun --partition=gpu --ntasks=1 --gpus=1 --mem-per-cpu=16000MB hist_cuda.out "$image")

        # Extract execution times from output
        hist_naive=$(echo "$gpu_output" | grep "Hist (naive):" | awk '{print $3}')
        hist_local=$(echo "$gpu_output" | grep "Hist (local mem):" | awk '{print $4}')
        cdf_naive=$(echo "$gpu_output" | grep "CDF (naive):" | awk '{print $3}')
        cdf_blelloch=$(echo "$gpu_output" | grep "CDF (blelloch):" | awk '{print $3}')

        hist_naive_times+=($hist_naive)
        hist_local_times+=($hist_local)
        cdf_naive_times+=($cdf_naive)
        cdf_blelloch_times+=($cdf_blelloch)

        # Store the result in the output file
        echo "$image, Run #$run: Hist (naive): $hist_naive ms, Hist (local mem): $hist_local ms, CDF (naive): $cdf_naive ms, CDF (blelloch): $cdf_blelloch ms" >> $output_file
    done

    # Report
    # Calculate averages of ALL times
    hist_naive_avg=$(echo "${hist_naive_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    hist_local_avg=$(echo "${hist_local_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    cdf_naive_avg=$(echo "${cdf_naive_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')
    cdf_blelloch_avg=$(echo "${cdf_blelloch_times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i} END {if (NF>0) print sum/NF; else print 0}')

    # Calculate speedups for different implementations
    if (( $(echo "$hist_naive_avg > 0" | bc -l) )); then
        hist_speedup=$(echo "scale=4; $hist_naive_avg / $hist_local_avg" | bc -l)
    else
        hist_speedup="N/A"
    fi

    if (( $(echo "$cdf_naive_avg > 0" | bc -l) )); then
        cdf_speedup=$(echo "scale=4; $cdf_naive_avg / $cdf_blelloch_avg" | bc -l)
    else
        cdf_speedup="N/A"
    fi

    # Extract the full image dimensions line from GPU output
    image_info=$(echo "$gpu_output" | grep "Image size:")

    # Write the summary results to the output file
    echo "-------------------------------- SUMMARY FOR $image --------------------------------" >> $output_file
    echo $image_info >> $output_file
    echo "AVERAGE TIMES (ALL RUNS):" >> $output_file
    echo "Histogram (naive): $hist_naive_avg ms" >> $output_file
    echo "Histogram (local mem): $hist_local_avg ms" >> $output_file
    echo "CDF (naive): $cdf_naive_avg ms" >> $output_file
    echo "CDF (blelloch): $cdf_blelloch_avg ms" >> $output_file
    echo "SPEEDUPS:" >> $output_file
    echo "Histogram naive vs local memory: $hist_speedup x faster" >> $output_file
    echo "CDF naive vs Blelloch: $cdf_speedup x faster" >> $output_file
    echo "--------------------------------------------------------------------------------" >> $output_file
    echo "" >> $output_file # add an empty line after each image's results
    echo "Tests for $image completed"

done

echo "All tests completed"