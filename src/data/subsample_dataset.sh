source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"

dataset_folder=/is/cluster/shared/datasets/is-rg-rml/laion400m/laion400m-data/
n=40000000000
worker_index=$1
num_workers=10


echo worker_index="$worker_index"
echo num_workers="$num_workers"
output_folder=/fast/pmayilvahanan/datasets/mix_2_14/
img_list=/is/cluster/fast/pmayilvahanan/clip_ood_part2/paths/mix_subsets/mix_2_14.npy
python3 subsample_dataset.py $worker_index $num_workers -d "$dataset_folder" -o "$output_folder" -i "$img_list" -n $n