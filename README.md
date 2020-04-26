# image_captioning_DL
# Source
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

# To run
## Install requirements
1. pip install -r requirements.txt

## Download pretrained glove word embeddings
2. chmod +x download_glove6B.sh
3. ./download_glove6B.sh

## Build glove dictionary for use during training
4. python build_glove

For the glove6B files, there are 4 different files of different dimensions:
50d, 100d, 200d and 300d. By default, running the above line would use the default dimension of 50

If you want to test out with the other 3 dimensions, please specify what embedding dimensions you prefer by using the --embed_size flag when running the 'build_glove.py' script

Example:
python build_glove --embed_size 100

If you change the embedding dimension size here, you will have to include the same flag and dimension size for 'train.py' and 'app.py' scripts when running them.

## Preprocessing
5. python build_vocab.py   
6. python resize.py

## Training
7. python train.py 

As mentioned earlier, if you decided to change the embedding dimensions from the default 50, please include the --embed_size flag

Example:
python train.py --embed_size 100

## GUI
8. python app.py

If you changed the default dimension size when building the glove dictionary, indicate the --embed_size flag

Example: python app.py --embed_size 100

9. select the folder where the images are stored for doing captioning prediction
