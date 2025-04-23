module load cudnn/8.4.1-cu11.6  # Trying different cuda version
echo "Loaded cudnn/8.7.0-cu11.x"
source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"
#dir=
#bs=
python3 style_classifer.eval --directory dr --batch_size bs
echo DONE.