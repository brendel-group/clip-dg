module load cudnn/8.4.1-cu11.6  # Trying different cuda version
echo "Loaded cudnn/8.7.0-cu11.x"
source /home/pmayilvahanan/.env/bin/activate
echo "activated source from /home/pmayilvahanan/.env/bin/activate"

python3 -m style_classifier.train --lr 0.1 --arch resnet18 --scheduler step

#python3 train.py --fourier --shift_spectrum \
#        --lr 0.01 --arch resnet50 --scheduler cosine
