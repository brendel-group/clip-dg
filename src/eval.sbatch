#!/bin/bash
#SBATCH --job-name=eval_style_classifier # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-20:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --mem=100G                 # Memory pool for all cores
#SBATCH --output=/mnt/qb/work/bethge/pmayilvahanan31/clip_ood_part2/jobs/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/pmayilvahanan31/clip_ood_part2/jobs/%x_%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU
##SBATCH --array=0-0               # start from '0'

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# Variable for run.sh
id=${SLURM_ARRAY_TASK_ID}
export id

# run the actual command
srun \
singularity exec \
--bind /mnt/qb/work/bethge/pmayilvahanan31/ \
--bind /mnt/qb/ \
--nv \
/mnt/qb/work/bethge/pmayilvahanan31/singularity_images/main_20231016.sif \
python3 src.style_classifer.eval --directory $1 --batch_size $2

echo DONE.