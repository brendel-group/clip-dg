echo "Usage: start_jupyter.sh IMAGE N_GPUS (PARTITION) (SINGULARITY_ARGS)"
if [ -z ${2+x} ]; then exit; fi
gpu="$2"
ram=$((30000*gpu))
if [ "$gpu" -gt 0 ]; then
        partition="${3:-gpu-2080ti}"
        cpu=$((8*gpu))
else
        partition="${3:-cpu-long}"
        cpu="2"
        gpu="0"
fi
echo "Running image $1 with ${gpu} GPUs and ${cpu} CPUs on partition ${partition}"
args=$4
srun $args  --job-name "main" --time=3-00:00 --cpus-per-task=$cpu --gres=gpu:$gpu --mem=50000M --ntasks=1 --nodes=1 --partition=$partition $args --pty singularity exec --nv --bind /scratch_local/ --bind /mnt/qb/ $1 /bin/bash -c "jupyter lab --ip=0.0.0.0 --no-browser"
