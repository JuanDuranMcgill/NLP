"""
This file contains all the models used in the linearization project of the IFT-6285 class

author: @Maxime Monrat
Created on Wed, Dec 15, 2021
"""
from os import name
from tensorflow.keras.layers import Layer, Bidirectional, LSTM, Dropout, Embedding, Dense, Activation, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
import util
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import util

def GenerateBatch(X, Y, vocabSize, batch_size=64, maxLen = 27):
    ''' 
    Generate a batch of data.
    Takes X and Y as lists of int and yields them batches by batches, converting Y to one-hot encoding on the go    

    '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_inputs = np.zeros((batch_size, maxLen),dtype='float32')
            decoder_inputs = np.zeros((batch_size, maxLen+2),dtype='float32')
            decoder_outputs = np.zeros((batch_size, maxLen+2, vocabSize),dtype='float32')

            for i, (input_text_seq, target_text_seq) in enumerate(zip(X[j:j+batch_size], Y[j:j+batch_size])):
                for t, word_index in enumerate(input_text_seq):
                    encoder_inputs[i, t] = word_index # encoder input seq

                for t, word_index in enumerate(target_text_seq):
                    decoder_inputs[i, t] = word_index
                    if (t>0) and (word_index<=vocabSize):
                        decoder_outputs[i, t-1, int(word_index-1)] = 1.

            yield([encoder_inputs, decoder_inputs], decoder_outputs)

def EmbeddingLayer(embedding_dict, word_to_index, pretrained = True):
    """
    Creates a Keras Embedding() layer and loads in pre-trained embedding vectors.
    
    Arguments:
        embedding_dict -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary
        pretrained -- boolean specifying if we are using pretrained embeddings (default: True)

    Returns:
        embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding
    emb_dim = embedding_dict["hat"].shape[0]      # dimensionality of embedding vectors
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = embedding_dict[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable= not pretrained)
    embedding_layer.build((None,))    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

class EncoderLSTM(Layer):
    def __init__(self, embedding_dict, word_to_index, n_a = 32, **kwargs):
        super(EncoderLSTM, self).__init__()

        self.embedding = EmbeddingLayer(embedding_dict,word_to_index)
        self.lstm = LSTM(n_a, return_sequences=False, return_state=True)
    
    def call(self, inputs, training=None):
        encoder_embeddings = self.embedding(inputs)
        X, state_h, state_c = self.lstm(encoder_embeddings)
        encoder_states = [state_h, state_c]
        return X, encoder_states

class EncoderBiLSTM(Layer):
    def __init__(self, embedding_dict, word_to_index, n_a = 32, dropout_rate=0.5, **kwargs):
        super(EncoderBiLSTM, self).__init__()

        self.embedding = EmbeddingLayer(embedding_dict,word_to_index)
        self.bilstm = Bidirectional(LSTM(n_a, return_sequences=True, return_state=False))
        self.lstm = LSTM(n_a, return_sequences=False, return_state=True)
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        encoder_embeddings = self.embedding(inputs)
        X = self.bilstm(encoder_embeddings)
        X = self.dropout(X)
        X, state_h, state_c = self.lstm(X)
        encoder_states = [state_h, state_c]
        
        return X, encoder_states

class EncoderCNN(Layer):
    def __init__(self, embedding_dict, word_to_index, n_s, dropout_rate=0.5, **kwargs):
        super(EncoderCNN, self).__init__()
        
        self.n_s=n_s
        self.embedding = EmbeddingLayer(embedding_dict,word_to_index)
        self.conv1 = Conv1D(1024,3, padding='same', activation='relu')
        self.conv2 = Conv1D(512,3, padding='same', activation='relu')
        self.conv3 = Conv1D(256,3, padding='same', activation='relu')
        self.pool = MaxPooling1D()
        self.pool2 = GlobalMaxPooling1D()
        self.flatten = Flatten()
        self.fc = Dense(n_s)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None):        
        encoder_embeddings = self.embedding(inputs)
        X = self.conv1(encoder_embeddings)
        X = self.conv2(X)
        X = self.pool(X)
        X = self.conv3(X)
        X = self.pool2(X)
        # X = self.flatten(X)
        # X = self.dropout(X)
        state_c = self.fc(X)
        state_h = self.fc(X)
        # state_c = tf.reshape(state_c, shape=[tf.shape(state_c)[0]*tf.shape(state_c)[1],tf.shape(state_c)[2]])
        # state_h = tf.reshape(state_h, shape=[tf.shape(state_h)[0]*tf.shape(state_h)[1],tf.shape(state_h)[2]])
        encoder_states = [state_h, state_c]
        
        return state_c, encoder_states

class DecoderLSTM(Layer):
    def __init__(self, embedding_dict, word_to_index, n_s = 64, dropout_rate=0.5, **kwargs):
        super(DecoderLSTM, self).__init__()

        vocabSize = len(word_to_index.keys())
        self.lstm = LSTM(n_s, return_sequences=True, return_state=True)
        self.embedding = EmbeddingLayer(embedding_dict,word_to_index)
        self.dense = Dense(vocabSize, activation='softmax')
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, encoder_states, training=None):
        decoder_embeddings = self.embedding(inputs)
        X, _, _ = self.lstm(decoder_embeddings, initial_state=encoder_states)
        X = self.dropout(X)
        X = self.dense(X)
        return X

class Seq2Seq(Model):
    """
    Architecture for the sequence-to-sequence Model used in the linearization task.
    This model is comprised of an encoder and a decoder, and uses teacher forcing for training.

    The encoder type can be chosen via the 'encoder_type' argument. There are 3 types available:
        'lstm' -- a simple LSTM RNN many-to-one
        'bilstm' -- a BiLSTM RNN
        'CNN' -- a simple 1D CNN
    
    the encoder outputs and states can be accessed after training via the 'encoder_outputs' and 'encoder_states' properties

    Arguments:        
        embedding_dict -- the dictionnary mapping every word to its vectorial representation
        word_to_index -- the dictionnary mapping every word in the vocabulary to its index (used in the embedding layer)
        max_words -- int, the maximum size, in words, of the sentences/bags of words (default=27)
        encoder_type -- string, 'lstm' 'bilstm' or 'cnn'. Specifies the type of encoder to be used
        n_a -- int, number of hidden units in the (bi)lstm layer of the encoder (default=32)
        n_s -- int, number of hidden units in the lstm layer of the decoder (default=64)
        name -- string, name of the model
    """
    def __init__(
        self,
        embedding_dict,
        word_to_index,
        max_words=27,
        encoder_type='lstm',
        n_a=32,
        n_s=64,
        name="seq2seq",
        **kwargs
    ):
        super(Seq2Seq, self).__init__(name=name, **kwargs)
        self.inputs = [Input((max_words,)), Input((max_words,))]

        # Check the desired type of encoding
        if encoder_type == 'bilstm':
            self.encoder = EncoderBiLSTM(embedding_dict, word_to_index, n_a)
            self.encoder_type = 'bilstm'
        elif encoder_type == 'cnn':
            self.encoder = EncoderCNN(embedding_dict, word_to_index, n_s)
            self.encoder_type = 'cnn'
        else:
            self.encoder = EncoderLSTM(embedding_dict, word_to_index,n_a)
            self.encoder_type = 'lstm'

        self.decoder = DecoderLSTM(embedding_dict, word_to_index, n_s)
    
    def call(self, inputs):        
        encoder_outputs, encoder_states = self.encoder(inputs[0])
        X = self.decoder(inputs[1], encoder_states)
        
        ## Save values for inference ##
        self.encoder_outputs = encoder_outputs
        self.encoder_states = encoder_states
        return X

def GetSamplingModels(model, n_s, maxLen, embedding_dict, word_to_index):
    """
    This function extracts the encoder and decoder part of a pre-trained seq2seq model.

    Arguments:
        model -- the previously trained seq2seq model
        n_s -- number of hidden states for the decoder's LSTM 
        maxLen -- max length of the data
        embedding_dict -- the embedding dictionary
        word_to_index -- the dictonary mapping every word to its index in the vocabulary
    
    Returns:
        encoder_model -- the encoder model, inputs: (maxLen,) outputs: (n_s,)
        decoder_model -- the decoder model, inputs: [(maxLen,),(2,n_s)] ourputs: [(maxLen,),(2,n_s)]
    """

    vocabSize = len(word_to_index.keys())
    encoder_inputs = Input((maxLen,))
    _, encoder_states = model.encoder(encoder_inputs)
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = Input((maxLen,))
    decoder_state_input_h = Input(shape=(n_s,))
    decoder_state_input_c = Input(shape=(n_s,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embeddings = EmbeddingLayer(embedding_dict,word_to_index)(decoder_inputs)
    X, state_h, state_c = LSTM(n_s, return_sequences=True, return_state=True)(decoder_embeddings, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = Dense(vocabSize, activation='softmax')(X)
    
    decoder_model = Model([decoder_inputs, decoder_states_inputs], [decoder_outputs, decoder_states])
    
    return encoder_model, decoder_model
