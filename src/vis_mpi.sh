module load cudnn/8.4.1-cu11.6  # Trying different cuda version
echo "Loaded cudnn/8.7.0-cu11.x"
source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"
#ids=
#img_dir=
#logits=
#preds=
#sample=
#n_imgs=

python3 -m visualization.main --ids ids --img_dir img_dir --logits logits --preds preds --sample sample --n_imgs n_imgs