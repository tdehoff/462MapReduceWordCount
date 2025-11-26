#!/bin/bash
#SBATCH -J exam_1 #Job name
#SBATCH -A ISAAC-UTK0414
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1#--ntasks is used when we want to define total number of processors
#SBATCH --cpus-per-task=2
#SBATCH --output=debug1.log    
#SBATCH --error=eprob1.log
#SBATCH --qos=short

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TEST_DIR="raw_text_input"

args=()

for file in "$TEST_DIR"/*; do
    args+=("$file")
done

module load openmpi/4.1.5-gcc

srun -n 4 ./hybrid "${args[@]}" > hybrid_out.txt


