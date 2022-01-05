#!/usr/bin/env python
# -*- coding: utf-8 -*-

import models
import util
import math
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from datetime import datetime

### Hyper-Parameters ###
MAXLEN = 25
LEARNING_RATE = 0.001
ENC_UNITS = 64
DEC_UNITS = 64
BATCH_SIZE = 10
EPOCHS = 30
########################

maxLen = MAXLEN + 2 # Adding 2 to max sentence length because of the <eos> <bos> tags
n_a = ENC_UNITS
n_s = DEC_UNITS


print('Importing training data...' )
# Get the training data
X_train, Y_train, X_test, Y_test = util.GetData('data/1BW.train', 'data/1BW.ref', 'data/devdata.test', 'data/devdata.ref')

# Get the vocabulary dictionnary and corresponding embeddings
glove_dict, word_to_index, index_to_word = util.LoadVectors('data/glove_small.txt')
vocabSize = len(word_to_index.keys())

# Convert input data to indices
X_train_int = util.Sentences2Indices(X_train, word_to_index, maxLen)
Y_train_int = util.Sentences2Indices(Y_train, word_to_index, maxLen)
X_test_int = util.Sentences2Indices(X_test, word_to_index, maxLen)
Y_test_int = util.Sentences2Indices(Y_test, word_to_index, maxLen)

modelType = None
while (modelType == None):
    modelTypeInput = input('Model to train?\n\t1 - LSTM encoder\n\t2 - Bi-LSTM encoder\n\t3 - CNN encoder\n')
    if modelTypeInput == '3':
        modelType = 'cnn'
    elif modelTypeInput == '2':
        modelType = 'bilstm'
    elif modelTypeInput == '1':
        modelType = 'lstm'
    else:
        print("Alors là n'importe quoi. (il faut répondre 1, 2 ou 3. Pas 'banane')")
        modelType = None


# model = models.Seq2Seq(
#     max_words=maxLen, 
#     embedding_dict=glove_dict, 
#     word_to_index=word_to_index, 
#     encoder_type=modelType, 
#     n_a=ENC_UNITS,
#     n_s=DEC_UNITS)

# model.build((maxLen,))
# model.summary()
# optimizer = Adam(learning_rate=LEARNING_RATE)
# model.compile(optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

model = load_model('models/seq2seq_cnn2021-12-19_13-26')

## !! IMPORTANT !! ## use KeyBoardInterrupt to stop early
saved = False
history = None
try:
    history = model.fit_generator(
        generator= models.GenerateBatch(X_train_int, Y_train_int, vocabSize, BATCH_SIZE),
        steps_per_epoch= math.ceil(len(X_train_int)/BATCH_SIZE),
        epochs=EPOCHS,
        verbose=1,
        validation_data= models.GenerateBatch(X_test_int, Y_test_int, vocabSize, batch_size=BATCH_SIZE),
        validation_steps=math.ceil(len(X_test_int)/BATCH_SIZE),
        workers=1)

except KeyboardInterrupt:
    fileName = 'seq2seq_'+ modelType + datetime.today().strftime('%Y-%m-%d_%H-%M')
    savePath = 'models\\' + fileName
    model.save(savePath)
    with open(savePath + '_HISTORY', 'wb') as historyfile, open(savePath + '_PICKLED', 'wb') as modelfile:
        pickle.dump(history, historyfile)
        pickle.dump(model, modelfile)
    saved = True

if saved == False:
    fileName = 'seq2seq_'+ modelType + datetime.today().strftime('%Y-%m-%d_%H-%M')
    savePath = 'models\\' + fileName
    model.save(savePath)
    with open(savePath + '_HISTORY', 'wb') as historyfile, open(savePath + '_PICKLED', 'wb') as modelfile:
        pickle.dump(history, historyfile)
        pickle.dump(model, modelfile)
    saved = True