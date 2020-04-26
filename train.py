import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
import bcolz
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
import shutil
from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def train(train_data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, train_encoder):
    decoder.train()
    encoder.train()
    # Train the models
    total_step = len(train_data_loader)
    
    losses = 0
    for i, (images, captions, lengths, sentences) in enumerate(train_data_loader):
        
        # if i >= 20:
        #     break

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        losses += loss.item()
        decoder.zero_grad()
        encoder.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        if train_encoder:
            encoder_optimizer.step()
        else:
            decoder_optimizer.step()

        # Print log info
        if i % args.log_step == 0:
            print(f'Training Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}') 

    losses /= len(train_data_loader)

    return losses
    

def validate(val_data_loader, encoder, decoder, criterion, vocab, epoch):
    decoder.eval()
    encoder.eval()
    
    val_total_step = len(val_data_loader)

    ref = []
    hypo = []
    losses = 0
    for i, (images, captions, lengths, target_caps) in enumerate(val_data_loader):

        # if i >= 20:
        #     break

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Generate an caption from the image
        feature = encoder(images)
        outputs = decoder(feature, captions, lengths)
        loss = criterion(outputs, targets)
        losses += loss.item()

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
            print(f'Validation Epoch [{epoch}/{args.num_epochs}], Step [{i}/{len(val_data_loader)}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')

    losses /= len(val_data_loader)
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
    return score_dict, losses


def main(args):
    checkpoint = True
    
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
        
    if args.with_glove:
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

        # Build the models
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNNGlove(args.hidden_size, weights_matrix, args.num_layers).to(device)
    else:
        # Build models normally 
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

        
            
    # Build data loader
    train_data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    
    val_data_loader = get_loader(args.val_image_dir, args.val_caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 


    if not args.reset_training:
        if isfile(os.path.join(args.model_path, 'best_encoder.ckpt')):
            encoder.load_state_dict(torch.load(os.path.join(args.model_path, 'best_encoder.ckpt')))
            print('Encoder weights loaded!')
        else:
            print('Weights file for encoder does not exist. Encoder will be initialized with default values.')

        if isfile(os.path.join(args.model_path, 'best_decoder.ckpt')):
            decoder.load_state_dict(torch.load(os.path.join(args.model_path, 'best_decoder.ckpt')))
            print('Decoder weights loaded!')
        else:
            print('Weights file for decoder does not exist. Decoder will be initialized with default values.')

        if isfile(os.path.join(args.model_path, 'last_best_bleu4.npy')):
            temp = np.load(os.path.join(args.model_path, 'last_best_bleu4.npy'), allow_pickle='TRUE').item()
            best_bleu4 = temp['best_bleu4']
            train_encoder = temp['train_encoder']
            print(f'Previous best bleu4 score: {best_bleu4}, training_encoder: {train_encoder}')
        else:
            best_bleu4 = 0
            train_encoder = False
    else:
        best_bleu4 = 0
        train_encoder = False

    best_epoch = 0

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    encoder_optimizer = torch.optim.Adam(params, lr=args.encoder_learning_rate)
    decoder_optimizer = torch.optim.Adam(params, lr=args.decoder_learning_rate)
    
    train_losses = []
    val_losses = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    cider_scores = []
    rouge_scores = []
    for epoch in range(1, args.num_epochs+1):
    
        train_loss = train(train_data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, train_encoder)
        score_dict, val_loss =validate(val_data_loader, encoder, decoder, criterion, vocab, epoch)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleu1_scores.append(score_dict['Bleu_1'])
        bleu2_scores.append(score_dict['Bleu_2'])
        bleu3_scores.append(score_dict['Bleu_3'])
        bleu4_scores.append(score_dict['Bleu_4'])
        cider_scores.append(score_dict['CIDEr'])
        rouge_scores.append(score_dict['ROUGE_L'])

        # Check if there was an improvement
        bleu4_score = score_dict['Bleu_4']
        print(f'Last best score {best_bleu4}, at epoch {best_epoch}')
        if bleu4_score > best_bleu4:
            best_bleu4 = bleu4_score
            best_epoch = epoch
            print(f'New best score {best_bleu4}, at epoch {best_epoch}')
            torch.save(decoder.state_dict(), os.path.join(args.model_path, 'best_decoder.ckpt'))
            torch.save(encoder.state_dict(), os.path.join(args.model_path, 'best_encoder.ckpt'))
            np.save(os.path.join(args.model_path, 'last_best_bleu4.npy'), {'best_bleu4': best_bleu4, 'train_encoder': train_encoder})

        else:
            if train_encoder:
                train_encoder = False
                print('No impovement in Bleu4 score. Switching from training Encoder to Decoder')
            else:
                train_encoder = True
                print('No impovement in Bleu4 score. Switching from training Decoder to Encoder')
            
            np.save(os.path.join(args.model_path, 'last_best_bleu4.npy'), {'best_bleu4': best_bleu4, 'train_encoder': train_encoder})

   #########################################################################################
        
    plot_loss_graph(args.num_epochs, train_losses, val_losses, init_folder)
    plot_score_graph(args.num_epochs, bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, cider_scores, rouge_scores, init_folder)


def plot_loss_graph(epoch, training_losses, val_losses, init_folder):
    epoch_list = np.arange(1, epoch + 1)
    plt.figure(figsize=(20,5))
    plt.plot(epoch_list, training_losses, label = "Training loss")
    plt.plot(epoch_list, val_losses, label = "Validation loss")
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = "upper right")
    plt.title('Training and Validation Losses')
    path = str(init_folder) + "/loss.png"
    plt.savefig(path)
    # plt.show()

def plot_score_graph(epoch, bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, cider_scores, rouge_scores, init_folder):
    epoch_list = np.arange(1, epoch + 1)
    plt.figure(figsize=(20,5))
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.plot(epoch_list, bleu1_scores, label = "bleu1 score")
    plt.plot(epoch_list, bleu2_scores, label = "bleu2 score")
    plt.plot(epoch_list, bleu3_scores, label = "bleu3 score")
    plt.plot(epoch_list, bleu4_scores, label = "bleu4 score")
    plt.plot(epoch_list, cider_scores, label = "cider score")
    plt.plot(epoch_list, rouge_scores, label = "rouge score")
    plt.legend(loc = "upper right")
    plt.title('Scores')
    path = str(init_folder) + "/score.png"
    plt.savefig(path)
    # plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--val_image_dir', type=str, default='data/val_resized2014', help='directory for validation resized images')
    parser.add_argument('--caption_path', type=str, default='/home/jovyan/datasets/coco2014/trainval_coco2014_captions/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='/home/jovyan/datasets/coco2014/trainval_coco2014_captions/captions_val2014.json', help='path for val annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--reset_training', type=bool , default=False, help='continue training from last best saved weights')
    
    # Glove path
    parser.add_argument('--with_glove', type=bool, default=True, help='set to false if using old decoder model')
    parser.add_argument('--glove_path', type=str , default='glove_data', help='give the path to glove directory')    
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=50, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--encoder_learning_rate', type=float, default=1e-4)
    parser.add_argument('--decoder_learning_rate', type=float, default=4e-4)
    args = parser.parse_args()
    print(args)
    main(args)