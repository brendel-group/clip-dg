source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"

output_folder=/is/cluster/fast/pmayilvahanan/datasets/mix_2_14/  # add '/' in the end

python3 rename_tars.py -o "$output_folder"