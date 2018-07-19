import pickle
import argparse
import numpy as np
from model import Options, HierarchicalSeq2Seq

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type = str, default = 'data',
                    help = 'the directory to the training data')
parser.add_argument('--num_epochs', type = int, default = 5,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'the batch size')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'the learning rate')
parser.add_argument('--beam_width', type = int, default = 10,
                    help = 'the beam width when decoding')
parser.add_argument('--embedding_size', type = int, default = 256,
                    help = 'the size of word embeddings')
parser.add_argument('--num_hidden_layers', type = int, default = 1,
                    help = 'the number of hidden layers')
parser.add_argument('--num_hidden_units', type = int, default = 1024,
                    help = 'the number of hidden units')
parser.add_argument('--save_path', type = str, default = 'model/model.ckpt',
                    help = 'the path to save the trained model to')
parser.add_argument('--restore_path', type = str, default = 'model/model.ckpt',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore', type = bool, default = False,
                    help = 'whether to restore from a trained model')
parser.add_argument('--predict', type = bool, default = False,
                    help = 'whether to enter predicting mode')
args = parser.parse_args()

def read_data(data_path):
    enc_x = np.load('{}/enc_x.npy'.format(data_path))
    dec_x = np.load('{}/dec_x.npy'.format(data_path))
    dec_y = np.load('{}/dec_y.npy'.format(data_path))
    dialog_lens = np.load('{}/dialog_lens.npy'.format(data_path))
    enc_x_lens = np.load('{}/enc_x_lens.npy'.format(data_path))
    dec_x_lens = np.load('{}/dec_x_lens.npy'.format(data_path))
    with open('{}/vocabulary.pickle'.format(data_path), 'rb') as file:
        vocabulary = pickle.load(file)
    with open('{}/vocabulary_reverse.pickle'.format(data_path), 'rb') as file:
        vocabulary_reverse = pickle.load(file)
    return enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens, vocabulary, vocabulary_reverse

if __name__ == '__main__':
    enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens, vocabulary, vocabulary_reverse = read_data(args.data_path)
    max_dialog_len = enc_x.shape[1]
    max_utterance_len = enc_x.shape[2]

    options = Options(num_epochs = args.num_epochs,
                      batch_size = args.batch_size,
                      learning_rate = args.learning_rate,
                      beam_width = args.beam_width,
                      vocabulary_size = len(vocabulary),
                      embedding_size = args.embedding_size,
                      num_hidden_layers = args.num_hidden_layers,
                      num_hidden_units = args.num_hidden_units,
                      max_dialog_len = max_dialog_len,
                      max_utterance_len = max_utterance_len,
                      go_index = vocabulary['<go>'],
                      eos_index = vocabulary['<eos>'])
    model = HierarchicalSeq2Seq(options)

    if args.predict:
        model.restore(args.restore_path)
        def initialize_input():
            input_enc_x = np.zeros((args.batch_size, max_dialog_len, max_utterance_len), dtype = np.int32)
            input_dialog_lens = np.zeros(args.batch_size, dtype = np.int32)
            input_enc_x_lens = np.zeros((args.batch_size, max_dialog_len), dtype = np.int32)
            return input_enc_x, input_dialog_lens, input_enc_x_lens
        input_enc_x, input_dialog_lens, input_enc_x_lens = initialize_input()
        num_turns = 0
        while num_turns < max_dialog_len:
            # Get the post from the user and store it.
            post = input('Input: ')
            if post == '<new>':
                input_enc_x, input_dialog_lens, input_enc_x_lens = initialize_input()
                num_turns = 0
                continue
            if post == '<exit>':
                break
            words = post.lower().split()
            if len(words) > max_utterance_len - 2:
                # Truncate the post if its length exceeds the maximum.
                words = words[:(max_utterance_len - 2)]
            word_indices = [vocabulary['<go>']]
            for word in words:
                if word in vocabulary:
                    word_indices.append(vocabulary[word])
                else:
                    word_indices.append(vocabulary['<unk>'])
            word_indices.append(vocabulary['<eos>'])
            input_enc_x[0,num_turns,:len(word_indices)] = word_indices
            input_dialog_lens[0] += 1
            input_enc_x_lens[0,num_turns] = len(word_indices)
            num_turns += 1

            # Get the predicted response from the model and store it.
            response = model.predict(input_enc_x, input_dialog_lens, input_enc_x_lens)
            response_text = []
            input_enc_x[0,num_turns,0] = vocabulary['<go>']
            i = 0
            while i < max_utterance_len - 2 and i < len(response) and vocabulary_reverse[response[i]] != '<eos>':
                input_enc_x[0,num_turns,i+1] = response[i]
                response_text.append(vocabulary_reverse[response[i]])
                i += 1
            input_enc_x[0,num_turns,i+1] = vocabulary['<eos>']
            print('Robot:', ' '.join(response_text))
            input_dialog_lens[0] += 1
            input_enc_x_lens[0,num_turns] = i + 2
            num_turns += 1
    else:
        if args.restore:
            model.restore(args.restore_path)
        else:
            model.init_tf_vars()
        model.train(enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens)
        model.save(args.save_path)
