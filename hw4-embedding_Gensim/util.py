"""
This is the utilitary library used for the emmbedding of a corpus using gensim

Created on Tue Oct 12 10:58
@authors: Maxime Monrat, Juan Felipe Duran
"""
import re
import os
import gensim
import numpy as np
from gensim.models import Word2Vec as wv
from gensim.corpora import dictionary
from smart_open import open


def Tokenize(text):
    """
    Splits a text in a list of words, using an ensemble of preprocessing techniques:
    - lowercasing
    - removes stop words
    - removes special characters
    - removes noisy text starting with numbers
    - replaces numbers by <NUM>
    - replaces URLs by <URL>
    - removes words only made of vowels with more than 7 vowels (the longest vowels-only word in English is 7 letters long) 

    Arguments:
        text -- the input text in string format
    
    Returns:
        tokenized_text -- the tokenized text in a list of words
    """
    stoplist = set('for a of the and to in'.split())
    # Lowercase the text
    text = text.lower()
    # Replace URLs and email adresses
    text=re.sub("((http)?s?:\\/\\/)?\\w+\\.\\w+(\\.\\w+)?((\\/\\w+)*)(\\.\\w+)?|(\\S?)+\\@\\S+\\.\\w+","<URL>",text)
    # Replace laughs
    text=re.sub("[aA]?([Hh][aeiAEI]){2,}[hH]?", "<LAUGHS>", text)
    # Remove noisy words only composed of voyels
    text=re.sub("[aeiouyAEIOUY]{7,}", " ", text)
    # Remove noisy text of the shape "<NUM>something"
    text=re.sub("[0-9]+\\S+"," ",text)
    # Replace numbers
    text=re.sub("[0-9]+","<NUM>",text)
    # Remove special characters
    text=re.sub("\\W"," ",text)
    # Split words
    # Remove stop words
    return [word for word in re.split("\\s+",text) if word not in stoplist]

def GetFilesPaths(dir):
    """
    Returns the list of files contained in a directory
    """
    paths = []
    for dirpath, directories, files in os.walk(dir):
        for filename in files:
            paths.append(os.path.join(dirpath, filename))
    return paths

def TestModel(model, words, fileOut = True):
    """
    Test a model by outputing the 10 nearest words and their scores for each word in a list.

    Arguments:
        model -- the gensim model to be tested
        words -- the list of words to test
        fileOut -- a boolean. If True, saves the results in a file
    
    Returns:
        results -- a list containing all the results
    """
    results = []

    for word in words:
        if(model.wv.has_index_for(word)):
            res_list = np.array(model.wv.most_similar(word))
            res_string = ''
            for pair in res_list:
                res_string += str(pair[0]) + " " + '[' + str(round(float(pair[1]), 3)) + ']' + " "
            results.append(str(word) + '\t' + res_string + '\n')

    if(fileOut):
        with open('voisins-nom.txt', 'w') as f:
            list(map(lambda item : f.write("%s" % item), results))   

    return results

class Sentences:
    """
    This class is used to lazy-load the sentences from a large sliced corpus in the Word2Vec model

    Arguments:
        dirname -- the relative path to the directory containing the corpus slices
        number_of_slices -- the maximum number of slices to take into account (default is -1, which takes all the existing slices)
    """
    def __init__(self, dirname, number_of_slices = -1):
        self.path = dirname
        self.number_of_slices = number_of_slices
        self.slicesPaths = GetFilesPaths(dirname)        
        
    def __iter__(self):
        c = 0
        for slicePath in self.slicesPaths:
            c +=1
            with open(slicePath) as slice:
                for line in slice:
                    yield Tokenize(slice.read())
            if ((c >= self.number_of_slices) and (self.number_of_slices > 0)):
                break

            

    
