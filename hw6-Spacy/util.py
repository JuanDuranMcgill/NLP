"""
Utilitary library

Created on Tue Nov 13
@authors: Maxime Monrat, Juan Felipe Duran
"""
import os 
import numpy as np 
from smart_open import open


def Get_Files_Paths(dir):
    """
    Returns the list of files contained in a directory
    """
    paths = []
    if os.path.exists(dir):
        for dirpath, directories, files in os.walk(dir):
            for filename in files:
                paths.append(os.path.join(dirpath, filename))        
    else:
        print('ERROR -- Missing Directory : ' + dir + ' cannot be reached.')
    
    return paths

def Get_Sentences(corpus, number_of_sentences):
    """ 
    Get a specified amount of sentences as a np.array of shape (number_of_sentences,) from a corpus object

    Arguments :
        corpus -- A corpus object containing all the slices of our corpus
        number_of_sentences -- int, number of desired sentences.
    
    Returns:
        sentences -- np array of shape (number_of_sentences,) containing all the sentences
    """
    sentences = np.zeros((number_of_sentences,), dtype=object)
    i = 0
    for line in corpus:

        if (i >= number_of_sentences):
            break
        sentences[i] = line
        i += 1
    if (i < number_of_sentences):
        print("WARNING -- the number of sentences asked exceeds the number of sentences in the corpus.\n" + i + " sentences yielded.")
    return sentences

       

class Corpus:
    """
    This class is used to lazy-load the sentences from a large sliced corpus in the Word2Vec model

    Arguments:
        dirname -- the relative path to the directory containing the corpus slices
        number_of_slices -- the maximum number of slices to take into account (default is -1, which takes all the existing slices)
    """
    def __init__(self, dirname, number_of_slices = -1):
        self.path = dirname
        self.number_of_slices = number_of_slices
        self.slicesPaths = Get_Files_Paths(dirname)        
        
    def __iter__(self):
        c = 0
        for slicePath in self.slicesPaths:
            c +=1
            with open(slicePath, encoding='utf-8') as slice:
                for line in slice:
                    yield line
            if ((c >= self.number_of_slices) and (self.number_of_slices > 0)):
                break