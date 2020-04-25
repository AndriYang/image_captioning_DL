import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_val import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
import shutil
from os import listdir
from os.path import isfile, join

from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    init_folder = 'results'
    
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    else:
        shutil.rmtree(init_folder)
        os.makedirs(init_folder)
        
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    ref = []
    hypo = []

    for i, (images, captions, lengths, target_caps) in enumerate(data_loader):
        # Generate an caption from the image
        images = images.to(device)
        feature = encoder(images)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        batch_ref = []
        batch_hypo = []
        for j, sentence_ids in enumerate(sampled_ids):
            sampled_caption = []
            for word_id in sentence_ids[1:]:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else: 
                    sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)
            batch_ref.append([sentence])
            batch_hypo.append([target_caps[j]])
        ref += batch_ref
        hypo += batch_hypo

        # Print log info
        if i % args.log_step == 0:
            print(f'Step [{i}/{total_step}]')
    
    #Bleu score format
    # ref = [['hi how are you'], ['not good']]
    # hypo = [['hi how are you'], ['not so good']]

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr"),
        (Rouge(), "ROUGE_L")
    ]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    score_dict = dict(zip(method,  score))

    print(score_dict)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/val_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../../../../../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    # Find the latest encoder and decoder
    model_root = 'models'
    files = [f for f in listdir(model_root) if isfile(join(model_root, f))]
    latest = int(len(files)/6)
    parser.add_argument('--encoder_path', type=str, default=f'models/encoder-{latest}-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=f'models/decoder-{latest}-3000.ckpt', help='path for trained decoder')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)