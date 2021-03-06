import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import bcolz
import os
from os import listdir
from os.path import isfile, join
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, DecoderRNNGlove
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    if args.with_glove == 'True':
        # Get glove pickles
        glove_path = args.glove_path

        vectors = bcolz.open(f'{glove_path}/6B.{args.embed_size}.dat')[:]
        words = pickle.load(open(f'{glove_path}/6B.{args.embed_size}_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.{args.embed_size}_idx.pkl', 'rb'))
        glove = {w: vectors[word2idx[w]] for w in words}

        # Get weights matrix
        weights_matrix = np.zeros((len(vocab), args.embed_size))
        words_found = 0

        # We compare the vocabulary from the built vocab, and the glove word vectors
        for i in range(len(vocab)):
            try: 
                word = vocab.idx2word[i]
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(args.embed_size, ))

        # Build models
        encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNNGlove(args.hidden_size, weights_matrix, args.num_layers)
    else:
        encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word != '<start>' and word != '<end>':
            sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    pickle.dump( sentence, open( "save.p", "wb" ) )
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    
    parser.add_argument('--encoder_path', type=str, default=f'models/best_encoder.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=f'models/best_decoder.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Glove path
    parser.add_argument('--with_glove', type=str, default='True', help='set to false if using old decoder model')
    parser.add_argument('--glove_path', type=str , default='glove_data', help='give the path to glove directory')    
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=50, help='dimension of glove word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)