import numpy as np
import pandas as pd
import tensorflow as tf
import util
import models
import itertools
import math
from progress.bar import Bar
from tensorflow.keras.layers import Layer, Bidirectional, LSTM, Dropout, Embedding, Dense, Activation, ActivityRegularization, Lambda, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam 
from tensorflow.python.keras import regularizers
from util import SentVariations, SortSentenceV1, GetLeastPerplexSent
from tokenizer import TextCleanup

def GetSamplingModels(model, n_s, maxLen, embedding_dict, word_to_index):
    vocabSize = len(word_to_index.keys())
    encoder_inputs = Input((maxLen,))
    _, encoder_states = model.encoder(encoder_inputs)
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = Input((maxLen,))
    decoder_state_input_h = Input(shape=(n_s,))
    decoder_state_input_c = Input(shape=(n_s,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embeddings = models.EmbeddingLayer(embedding_dict,word_to_index)(decoder_inputs)
    X, state_h, state_c = LSTM(n_s, return_sequences=True, return_state=True)(decoder_embeddings, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = Dense(vocabSize, activation='softmax')(X)
    
    decoder_model = Model([decoder_inputs, decoder_states_inputs], [decoder_outputs, decoder_states])
    
    return encoder_model, decoder_model

def OneStepBeamSearch(decoder_model, word, score, states, seq_voc, k):
    out_words = []
    out_scores = []
    out_states = []
    out_vocabs = []
    
    output_probs, output_states = decoder_model.predict([word, states]) # next word probs
    output_scores = tf.math.log(output_probs) # next word log(probs)
    seq_voc_out_scores = np.array(output_scores)[:,:,seq_voc.astype(np.int32)][0,0,:] # only consider words in the seq vocab
    best_k_next_in_seq_voc = np.argpartition(seq_voc_out_scores, -k)[-k:] # take the k best choices
    # for each beam:
    #   create a new seq with appended new index
    #   Compute new score
    #   Remove word from seq vocab
    for i in range(k):
        sampled_word_seq_index = best_k_next_in_seq_voc[i]
        sampled_word_index = seq_voc[sampled_word_seq_index]
        new_score = score + seq_voc_out_scores[sampled_word_seq_index]
        out_words.append(np.array([[sampled_word_index]]))
        out_scores.append(new_score)
        out_vocabs.append(np.array(np.delete(seq_voc, sampled_word_seq_index)))
        out_states.append(output_states)

    return out_words, out_scores, out_states, out_vocabs

def BeamSearch(decoder_model, branches, branchesScores, thisBranch, init_word, init_score, init_states, init_vocab, k = 1):
    
    # Check if there is enough remaining words
    if init_vocab.size <= k:
        k = init_vocab.size
    out_words, out_scores, out_states, out_vocabs =  OneStepBeamSearch(decoder_model, init_word, init_score, init_states, init_vocab, k)
    for i in range(k):
        thisBranch.extend(out_words[i])
        thisBranchScore = out_scores[i]
        if init_vocab.size < 2:
            branches.append(thisBranch)
            branchesScores.append(thisBranchScore)
            break
        else:
            branches, branchesScores = BeamSearch(decoder_model, branches, branchesScores, thisBranch, out_words[i], thisBranchScore, out_states[i], out_vocabs[i], k)

    return branches, branchesScores
    

def DecodeSequence(input_seq, encoder_model, decoder_model, word_to_index, maxLen, k = 1):
    realLen = input_seq[input_seq != 0].size
    beams = []
    scores = []
    init_word = np.array(word_to_index['<bos>']).reshape(1,1)

    # Encode the input as state vectors.
    input_seq = input_seq.reshape(1,maxLen).astype(np.int32)
    states_values = encoder_model.predict(input_seq)
    
    beams, scores = BeamSearch(decoder_model, beams, [], [], init_word, 1, states_values, input_seq[input_seq != 0], k)
    return beams[np.argmax(scores)], beams, scores

def Sequence2Sentence(seq, index_to_word):
    sent = ''
    for n in seq:
        if n[0] != 0:
            sent = sent + ' ' + index_to_word[n[0]]
    return sent.replace(" <eos>", '').replace(" <bos>", '').strip()

maxLen = 25 + 2 
glove_dict, word_to_index, index_to_word = util.LoadVectors('data\\glove.txt')


with open('data\\euro.test', 'r', encoding='utf-8') as feuro, open('data\\hans.test', 'r', encoding='utf-8') as fhans, open('data\\news.test', 'r', encoding='utf-8') as fnews:
        X_euro = np.array(['<bos> ' + line.strip() + ' <eos>' for line in feuro])
        X_hans = np.array(['<bos> ' + line.strip() + ' <eos>' for line in fhans])
        X_news = np.array(['<bos> ' + line.strip() + ' <eos>' for line in fnews])

X_euro_int = util.Sentences2Indices(X_euro, word_to_index, maxLen)
X_hans_int = util.Sentences2Indices(X_hans, word_to_index, maxLen)
X_news_int = util.Sentences2Indices(X_news, word_to_index, maxLen)
Y_euro = []
Y_hans = []
Y_news = []
X_tests = [X_euro_int, X_hans_int, X_news_int]
Y_tests = [Y_euro, Y_hans, Y_news]

model = load_model('models\\seq2seq_bilstm2021-12-18_10-55')
encoder_type = 'bilstm'
encoder_model, decoder_model = models.GetSamplingModels(model, 64, maxLen, glove_dict, word_to_index)
euro_out = 'out\\euro_' + encoder_type
hans_out = 'out\\hans_' + encoder_type
news_out = 'out\\news_' + encoder_type
with open(euro_out, 'w', encoding='utf-8') as feuro_out, open(hans_out, 'w', encoding='utf-8') as fhans_out, open(news_out, 'w', encoding='utf-8') as fnews_out, Bar("Decoding lines", max=3000, suffix='%(percent)d%% - %(eta)ds') as progressBar:
    fout = [feuro_out, fhans_out, fnews_out]
    # for each test file
    for i in range(3):    
        # for each sentence
        for input_seq in X_tests[i]:
            seq, _, _ = DecodeSequence(input_seq, encoder_model, decoder_model, word_to_index, maxLen, k = 1) # Greedy search
            fout[i].write(Sequence2Sentence(seq, index_to_word) + '\n')
            progressBar.next()