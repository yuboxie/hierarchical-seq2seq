import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type = str, default = 'corpus/dialogues_text.txt',
                    help = 'path to the data file')
parser.add_argument('--max_utterance_len', type = int, default = 35,
                    help = 'maximum length of utterance')
parser.add_argument('--vocabulary_size', type = int, default = 10000,
                    help = 'vocabulary size')
parser.add_argument('--save_path', type = str, default = 'data',
                    help = 'the directory to save the data to')
args = parser.parse_args()

def read_data(file_path, max_utterance_len):
    '''
    Read the dialog text from the txt file.
    Each dialog is a list of all the utterances in it.
    Each utterance is converted to lowercase and tokenized into a list of tokens.
    Dialogs containing utterances of length greater than max_utterance_len are discarded.
    - Input:
        file_path (str): the path to the txt file.
        max_utterance_len (int): the maximum length of utterance.
    - Return:
        dialogs (list): the list of all the dialogs.
        frequencies (defaultdict): the number of occurrences of each token.
    '''
    dialogs = []
    frequencies = defaultdict(int)
    file = open(file_path, 'r', encoding = 'utf-8')
    for line in file:
        if line[-1] == '\n':
            line = line[:-1]
        dialog = line.split(' __eou__ ')
        dialog[-1] = dialog[-1][:-8]
        dialog = [utterance.lower().split() for utterance in dialog]
        if np.max([len(utterance) for utterance in dialog]) > max_utterance_len:
            continue
        for utterance in dialog:
            for token in utterance:
                frequencies[token] += 1
        dialogs.append(dialog)
    return dialogs, frequencies

def print_info(dialogs, frequencies):
    '''
    Print some useful information about the dialog data.
    - Input:
        dialogs (list): the list of all the dialogs.
        frequencies (defaultdict): the number of occurrences of each token.
    '''
    print('Total number of dialogs: {}'.format(len(dialogs)))
    print('Total vocabulary size: {}'.format(len(frequencies)))
    max_dialog_len = np.max([len(dialog) for dialog in dialogs])
    print('Max number of turns: {}'.format(max_dialog_len))
    max_utterance_len = np.max([len(utterance) for dialog in dialogs for utterance in dialog])
    print('Max number of tokens: {}'.format(max_utterance_len))

def construct_vocabulary(frequencies, vocabulary_size):
    '''
    Construct the vocabulary with the most frequent tokens.
    - Input:
        frequencies (defaultdict): the number of occurrences of each token.
        vocabulary_size (int): the maximum number of tokens to be included.
    - Return:
        vocabulary (dict): the vocabulary with keys as tokens and values as indices.
        vocabulary_reverse (dict): the reversed vocabulary.
    '''
    tokens = sorted(frequencies, key = frequencies.get, reverse = True)
    tokens = ['<pad>', '<go>', '<eos>', '<unk>'] + tokens[:vocabulary_size]
    vocabulary = dict(zip(tokens, range(len(tokens))))
    vocabulary_reverse = dict(zip(range(len(tokens)), tokens))
    return vocabulary, vocabulary_reverse

def convert_to_integer_representation(dialogs, vocabulary):
    '''
    Convert the dialog text to integer representation.
    Tokens not found in the vocabulary are replaced with <unk>.
    - Input:
        dialogs (list): the list of all the dialogs.
        vocabulary (dict): the vocabulary with keys as tokens and values as indices.
    - Return:
        enc_x (numpy array): inputs to the encoder.
        dec_x (numpy array): inputs to the decoder.
        dec_y (numpy array): targets of the decoder.
        dialog_lens (numpy array): length of the dialogs.
        enc_x_lens (numpy array): length of the encoder inputs.
        dec_x_lens (numpy array): length of the decoder inputs.
    '''
    max_dialog_len = np.max([len(dialog) for dialog in dialogs])
    max_utterance_len = np.max([len(utterance) for dialog in dialogs for utterance in dialog])
    enc_x = np.zeros((len(dialogs), max_dialog_len - 1, max_utterance_len + 2), dtype = np.int32)
    dec_x = np.zeros((len(dialogs), max_dialog_len - 1, max_utterance_len + 2), dtype = np.int32)
    dec_y = np.zeros((len(dialogs), max_dialog_len - 1, max_utterance_len + 2), dtype = np.int32)
    dialog_lens = np.array([len(dialog) - 1 for dialog in dialogs], dtype = np.int32)
    enc_x_lens = np.zeros((len(dialogs), max_dialog_len - 1), dtype = np.int32)
    dec_x_lens = np.zeros((len(dialogs), max_dialog_len - 1), dtype = np.int32)
    for i in range(len(dialogs)):
        utterance_lens = np.array([len(utterance) for utterance in dialogs[i]])
        enc_x_lens[i,:dialog_lens[i]] = utterance_lens[:-1] + 2
        dec_x_lens[i,:dialog_lens[i]] = utterance_lens[1:] + 1
        for j in range(len(dialogs[i])):
            utterance_indices = [vocabulary['<go>']]
            for word in dialogs[i][j]:
                if word in vocabulary:
                    utterance_indices.append(vocabulary[word])
                else:
                    utterance_indices.append(vocabulary['<unk>'])
            utterance_indices.append(vocabulary['<eos>'])
            utterance_len = len(utterance_indices)
            if j < len(dialogs[i]) - 1:
                enc_x[i,j,:utterance_len] = utterance_indices
            if j > 0:
                dec_x[i,j-1,:utterance_len-1] = utterance_indices[:-1]
                dec_y[i,j-1,:utterance_len-1] = utterance_indices[1:]
    return enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens

def save_data(enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens, vocabulary, vocabulary_reverse, save_path):
    '''
    Save the data to files.
    - Input:
        enc_x (numpy array): inputs to the encoder.
        dec_x (numpy array): inputs to the decoder.
        dec_y (numpy array): targets of the decoder.
        dialog_lens (numpy array): length of the dialogs.
        enc_x_lens (numpy array): length of the encoder inputs.
        dec_x_lens (numpy array): length of the decoder inputs.
        vocabulary (dict): the vocabulary with keys as tokens and values as indices.
        vocabulary_reverse (dict): the reversed vocabulary.
        save_path (str): the directory to save the data to.
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save('{}/enc_x.npy'.format(save_path), enc_x)
    np.save('{}/dec_x.npy'.format(save_path), dec_x)
    np.save('{}/dec_y.npy'.format(save_path), dec_y)
    np.save('{}/dialog_lens.npy'.format(save_path), dialog_lens)
    np.save('{}/enc_x_lens.npy'.format(save_path), enc_x_lens)
    np.save('{}/dec_x_lens.npy'.format(save_path), dec_x_lens)
    with open('{}/vocabulary.pickle'.format(save_path), 'wb') as file:
        pickle.dump(vocabulary, file)
    with open('{}/vocabulary_reverse.pickle'.format(save_path), 'wb') as file:
        pickle.dump(vocabulary_reverse, file)

if __name__ == '__main__':
    # Read the dialog text from the txt file.
    dialogs, frequencies = read_data(args.file_path, args.max_utterance_len)
    print_info(dialogs, frequencies)

    # Construct the vocabulary with the most frequent tokens.
    vocabulary, vocabulary_reverse = construct_vocabulary(frequencies, args.vocabulary_size)

    # Convert the dialog text to integer representation.
    enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens = convert_to_integer_representation(dialogs, vocabulary)

    # Save the data to files.
    save_data(enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens, vocabulary, vocabulary_reverse, args.save_path)
