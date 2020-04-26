import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import bcolz
import numpy as np
import os
import pickle
from build_vocab import Vocabulary

from torch.nn.utils.rnn import pack_padded_sequence

def main(args):
    glove_path = args.glove_path
    embed_size = args.embed_size
    
    glove_6B_dat = f'{glove_path}/6B.{embed_size}.dat'
    if not os.path.exists(glove_6B_dat):    
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=glove_6B_dat, mode='w')

        with open(f'{glove_path}/glove.6B.{embed_size}d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        vectors = bcolz.carray(vectors[1:].reshape((-1, embed_size)), rootdir=glove_6B_dat, mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{glove_path}/6B.{embed_size}_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{glove_path}/6B.{embed_size}_idx.pkl', 'wb'))

        print('glove pickle dumping done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--glove_path', type=str , default='glove_data', help='give the path to glove directory')    
    parser.add_argument('--embed_size', type=int , default=50, help='dimension of glove embedding vectors')
    
    args = parser.parse_args()
    print(args)
    main(args)