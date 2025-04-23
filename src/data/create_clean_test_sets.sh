source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"

dataset_name=dn-sketch
python3 create_clean_test_sets.py -d $dataset_name