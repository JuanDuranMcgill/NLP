"""
This program trains an embedding model using Gensim and Word2Vec

Created on Tue Oct 12 12:06
@authors: Maxime Monrat, Juan Felipe Duran
"""

import util
import numpy as np
import gensim, logging
from gensim.models import Word2Vec
from util import Sentences
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


sentences = Sentences('../Corpus1B/training-monolingual.tokenized.shuffled', number_of_slices=-1)
model = Word2Vec(sentences, min_count=1,negative=20, window=10,vector_size=300, workers=8)
#model.save('model1BW')

#model = Word2Vec.load('model1BW')
print(np.array(model.wv.most_similar('president')))
model.wv.evaluate_word_analogies('questions-words.txt')
with open('liste_mots_devoir4.txt', 'r') as f:
    testWords = f.read().split()
util.TestModel(model, testWords)