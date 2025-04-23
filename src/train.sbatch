#!/bin/bash
#SBATCH --job-name=style_classifier # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --time=2-23:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-v100    # Partition to submit to
#SBATCH --mem=50G                 # Memory pool for all cores
#SBATCH --output=/mnt/qb/work/bethge/pmayilvahanan31/clip_ood_part2/jobs/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/pmayilvahanan31/clip_ood_part2/jobs/%x_%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU

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
python3 -m src.style_classifer.train --fourier --shift_images --shift_spectrum
#python3 train.py --fourier --shift_images --shift_spectrum


echo DONE.