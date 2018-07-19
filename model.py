import numpy as np
import tensorflow as tf

class Options(object):
    '''Parameters used by the HierarchicalSeq2Seq model.'''
    def __init__(self, num_epochs, batch_size, learning_rate, beam_width, vocabulary_size, embedding_size,
                 num_hidden_layers, num_hidden_units, max_dialog_len, max_utterance_len, go_index, eos_index):
        super(Options, self).__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beam_width = beam_width

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.max_dialog_len = max_dialog_len
        self.max_utterance_len = max_utterance_len
        self.go_index = go_index
        self.eos_index = eos_index

class HierarchicalSeq2Seq(object):
    '''A hierarchical sequence to sequence model for multi-turn dialog generation.'''
    def __init__(self, options):
        super(HierarchicalSeq2Seq, self).__init__()

        self.options = options

        self.build_graph()
        self.session = tf.Session(graph = self.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session is closed.')

    def build_graph(self):
        print('Building the TensorFlow graph...')
        opts = self.options

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.enc_x = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_utterance_len, opts.batch_size])
            self.dec_x = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_utterance_len, opts.batch_size])
            self.dec_y = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_utterance_len, opts.batch_size])

            self.dialog_lens = tf.placeholder(tf.int32, shape = [opts.batch_size])
            self.enc_x_lens = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.batch_size])
            self.dec_x_lens = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.batch_size])

            embeddings = tf.Variable(tf.random_uniform([opts.vocabulary_size, opts.embedding_size], -1.0, 1.0))
            enc_x_embed = tf.nn.embedding_lookup(embeddings, self.enc_x)
            dec_x_embed = tf.nn.embedding_lookup(embeddings, self.dec_x)

            # The encoder RNN.
            with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
                # Define the encoder cell.
                enc_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)

                # Perform encoding.
                enc_states = []
                for i in range(opts.max_dialog_len):
                    _, enc_state = tf.nn.dynamic_rnn(cell = enc_gru_cell,
                                                     inputs = enc_x_embed[i,:,:,:],
                                                     sequence_length = self.enc_x_lens[i,:],
                                                     dtype = tf.float32,
                                                     time_major = True)
                    enc_states.append(enc_state)

            # The context RNN.
            with tf.variable_scope('context', reuse = tf.AUTO_REUSE):
                rnn_input = tf.reshape(tf.stack(enc_states), [opts.max_dialog_len, -1, opts.num_hidden_units])
                rnn_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                rnn_output, _ = tf.nn.dynamic_rnn(cell = rnn_gru_cell,
                                                  inputs = rnn_input,
                                                  sequence_length = self.dialog_lens,
                                                  dtype = tf.float32,
                                                  time_major = True)

            # The decoder RNN.
            with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
                # Define the decoder cell and the output layer.
                dec_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                output_layer = tf.layers.Dense(units = opts.vocabulary_size,
                                               kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1))

                # Define the beam search decoder (on the last utterance of the first example of the batch).
                start_tokens = tf.constant([opts.go_index], dtype = tf.int32)
                initial_state = tf.expand_dims(rnn_output[self.dialog_lens[0]-1,0,:], 0)
                initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier = opts.beam_width)
                beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = dec_gru_cell,
                                                                           embedding = embeddings,
                                                                           start_tokens = start_tokens,
                                                                           end_token = opts.eos_index,
                                                                           initial_state = initial_state,
                                                                           beam_width = opts.beam_width,
                                                                           output_layer = output_layer)

                # Perform beam search decoding.
                self.dec_output_beam, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = beam_search_decoder,
                                                                               output_time_major = True,
                                                                               impute_finished = False,
                                                                               maximum_iterations = opts.max_utterance_len)

                # Perform training decoding.
                dec_outputs_train = []
                for i in range(opts.max_dialog_len):
                    dec_output_train, _ = tf.nn.dynamic_rnn(cell = dec_gru_cell,
                                                            inputs = dec_x_embed[i,:,:,:],
                                                            sequence_length = self.dec_x_lens[i,:],
                                                            initial_state = rnn_output[i,:,:],
                                                            dtype = tf.float32,
                                                            time_major = True)
                    dec_output_train = tf.reshape(dec_output_train, [-1, opts.num_hidden_units])
                    dec_output_train = output_layer.apply(dec_output_train)
                    dec_output_train = tf.reshape(dec_output_train, [opts.max_utterance_len, opts.batch_size, opts.vocabulary_size])
                    dec_outputs_train.append(dec_output_train)

            # Compute loss function.
            logits = tf.reshape(tf.transpose(tf.stack(dec_outputs_train), perm = [0, 2, 1, 3]), [-1, opts.max_utterance_len, opts.vocabulary_size])
            targets = tf.reshape(tf.transpose(self.dec_y, perm = [0, 2, 1]), [-1, opts.max_utterance_len])
            weights = tf.cast(tf.sequence_mask(tf.reshape(self.dec_x_lens, [-1]), opts.max_utterance_len), tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
            self.optimizer = tf.train.AdamOptimizer(opts.learning_rate).minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def init_tf_vars(self):
        self.session.run(self.init)
        print('TensorFlow variables initialized.')

    def train(self, enc_x, dec_x, dec_y, dialog_lens, enc_x_lens, dec_x_lens):
        print('Start to train the model...')
        opts = self.options

        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size
        for epoch in range(opts.num_epochs):
            perm_indices = np.random.permutation(range(num_examples))
            for batch in range(num_batches):
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                batch_enc_x = np.transpose(enc_x[batch_indices,:,:], [1, 2, 0])
                batch_dec_x = np.transpose(dec_x[batch_indices,:,:], [1, 2, 0])
                batch_dec_y = np.transpose(dec_y[batch_indices,:,:], [1, 2, 0])
                batch_dialog_lens = dialog_lens[batch_indices]
                batch_enc_x_lens = np.transpose(enc_x_lens[batch_indices,:])
                batch_dec_x_lens = np.transpose(dec_x_lens[batch_indices,:])
                feed_dict = {self.enc_x: batch_enc_x,
                             self.dec_x: batch_dec_x,
                             self.dec_y: batch_dec_y,
                             self.dialog_lens: batch_dialog_lens,
                             self.enc_x_lens: batch_enc_x_lens,
                             self.dec_x_lens: batch_dec_x_lens}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict = feed_dict)
                print('Epoch {:03d} batch {:04d}: loss = {}'.format(epoch + 1, batch + 1, loss_val), flush = True)

    def save(self, save_path):
        print('Saving the trained model...')
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print('Restoring from a pre-trained model...')
        self.saver.restore(self.session, restore_path)

    def predict(self, enc_x, dialog_lens, enc_x_lens):
        dec_x = np.zeros(enc_x.shape, dtype = np.int32)
        dec_y = np.zeros(enc_x.shape, dtype = np.int32)
        dec_x_lens = np.zeros(enc_x_lens.shape, dtype = np.int32)
        feed_dict = {self.enc_x: np.transpose(enc_x, [1, 2, 0]),
                     self.dec_x: np.transpose(dec_x, [1, 2, 0]),
                     self.dec_y: np.transpose(dec_y, [1, 2, 0]),
                     self.dialog_lens: dialog_lens,
                     self.enc_x_lens: np.transpose(enc_x_lens),
                     self.dec_x_lens: np.transpose(dec_x_lens)}
        dec_output = self.session.run(self.dec_output_beam, feed_dict = feed_dict)
        return dec_output.predicted_ids[:,0,0]
