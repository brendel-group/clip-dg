source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"
line_number=$(( $1 + 1 ))  # Adjust line number to start from 1
script2_path=jobs_eval.sh

# Read the line from script2 and store arguments in variables
line=$(sed -n "${line_number}p" "$script2_path")
echo $line

# Store line to vars
read -r model_name checkpoints_dir cnn pretrained <<<"$line"

python3 zero_shot_eval.py -m "$model_name" -c "$checkpoints_dir" --cnn "$cnn"  -p "$pretrained"
echo DONE.