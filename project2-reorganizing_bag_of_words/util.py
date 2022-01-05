"""
Utilitary library

Created on Tue Nov 30
@authors: Maxime Monrat, Juan Felipe Duran, Nathan Migeon
"""
import os 
import numpy as np 
import pandas as pd
import kenlm
import itertools
from progress.bar import Bar
from itertools import permutations
from collections import Counter
from tokenizer import TextCleanup

## Random seed for numpy ##
NPRAND = np.random.default_rng(seed=42)

def BufCountLines(fname):
    """
    Counts lines in a very large file in a cheaply manner.
    """
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def OneHot(values, number_of_classes):
    """
    Convert a 1-D numpy array to its corresponding one-hot representation

    Arguments:
        values: a numpy array of shape (n_values,) containing the vector to convert
        number_of_classes: int, the number of classes to use for the one-hot encoding
    
    Returns:
        one_hot: a numpy array of shape (n_values, number_of_classes)
    """

    one_hot = np.zeros((values.size, number_of_classes))
    one_hot[np.arange(values.size), values] = 1

    return one_hot

def LoadVectors(embeddings_file):
    """
    loads the embedding vectors from a file
    the file text should be in the following format:
        word x1 x2 x3 ... xn
    where x is a scalar and n the dimension of the vector, and 'word' the corresponding word

    Arguments:
        corpus_file -- the file containing the dataset 

    Returns:
        vectors_dict -- a dictionnary mapping every word to their embedding
        word_to_index -- a dictionnary mapping every word to its index
        index_to_word -- the inverse dictionnary of word_to_index

    """
    lineCount = BufCountLines(embeddings_file)

    vectors_dict = {}
    index_to_word = {}
    index_to_word[0] = '<pad>'
       
    with open(embeddings_file, 'r', encoding="utf-8") as f, Bar("Reading the embedding file", max=lineCount, suffix='%(percent)d%% - %(eta)ds') as progressBar:
        i = 1 
        for line in f:
            vals = line.split()
            word = vals[0]
            vect = np.asarray(vals[1:], "float32")    
            vectors_dict[word] = vect
            index_to_word[i] = word
            i += 1
            progressBar.next()
        # Add custom tokens
        index_to_word[i] = '<unk>'
        index_to_word[i+1] = '<bos>'
        index_to_word[i+2] = '<eos>'
    
    # Initialize <pad> at 0 and the others at random numbers
    vectors_dict['<pad>'] = np.zeros(vectors_dict['hat'].shape)
    vectors_dict['<unk>'] = NPRAND.random(vectors_dict['hat'].shape)
    vectors_dict['<bos>'] = NPRAND.random(vectors_dict['hat'].shape)
    vectors_dict['<eos>'] = NPRAND.random(vectors_dict['hat'].shape)
    word_to_index = {v:k for k,v in index_to_word.items()}

    return vectors_dict, word_to_index, index_to_word

def GetData(ftrain, ftrainref, ftest, ftestref, max_train_examples = 50000):
    """
    Gets the training and testing data from the train and test/dev sets. Adds <bos> and <eos> tags.

    Arguments:
        ftrain -- training file
        ftrainref -- training reference file
        ftest -- test file
        ftestref -- test reference file
        max_train_examples -- maximum training examples to extract (default: 50'000)

    Returns:
        X_train -- the input data of the training set, numpy array of shape (m, 1)
        Y_train -- the output data of the training set, numpy array of shape (m, 1)
        X_test -- the input data of the test or dev set, numpy array of shape (m, 1)
        Y_test -- the output data of the test or dev set, numpy array of shape (m, 1)
        
    """
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    with open(ftrain, 'r', encoding='utf-8') as f_x_train, open(ftrainref, 'r', encoding='utf-8') as f_y_train, open(ftest, 'r', encoding='utf-8') as f_x_test, open(ftestref, 'r', encoding='utf-8') as f_y_test:
        X_train = np.array(list(itertools.islice(['<bos> ' + line.strip() + ' <eos>' for line in f_x_train], max_train_examples)))
        Y_train = np.array(list(itertools.islice(['<bos> ' + line.strip() + ' <eos>' for line in f_y_train], max_train_examples)))
        X_test = np.array(list(itertools.islice(['<bos> ' + line.strip() + ' <eos>' for line in f_x_test], max_train_examples)))
        Y_test = np.array(list(itertools.islice(['<bos> ' + line.strip() + ' <eos>' for line in f_y_test], max_train_examples)))
       
    return X_train, Y_train, X_test, Y_test

def Sentences2Indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    This is a necessary step in order to use the Embedding() layer in Keras. 
    
    Arguments:
        X -- array of sentences (strings), of shape (m, 1)
        embedding_dict -- a dictionary containing the each word mapped to its index
        max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
        
    """
    
    m = X.shape[0]                                   # number of training examples    
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        sentence_words = [w.lower() for w in X[i].split()]
        j = 0
        for w in sentence_words:
            # Replace unknown words
            if (w not in word_to_index.keys()):
                w = '<unk>'
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices

def GetLeastPerplexSent(sents, model):
    """
    Get the sentence with the minimum of perplexity from a list of sentences

    Arguments:
        sents -- the list of sentences in string format
        model -- the KenLM n-gram model

    Returns:
        outSent -- the output sentence
        perplexity -- the perplexity of the output sentence
    """
    perplexity = model.perplexity(sents[0])
    for sent in sents:
        perp = model.perplexity(sent)
        if perp <= perplexity:
            perplexity = perp
            outSent = sent
    
    return outSent, perplexity

def GetNgrams(words, model, o = 4):
    """
    Gets the most probable n-gram for each word in a list
    Arguments:
        words -- the list of words
        model -- the KenLM n-gram model
        o -- int, order of the n-gram (default: 4)

    Returns:
        ngrams -- list of str, all the most probable n-grams for each word of the dictionnary
    """
    ngrams = []
    ngrams_current = []
    print(words)
    for word in words:
        # Get all possible permutations of order o
        perms = permutations(words, o)
        ngrams_current.clear()
        
        # For each word, get the permutations starting by it
        for perm in perms:            
            if str(word) == str(perm[0]):
                ngrams_current.append(' '.join(perm))
        # Get the least perplex ngram
        ngram, _ = GetLeastPerplexSent(ngrams_current, model)
        ngrams.append(ngram)

    return ngrams

def SentVariations(sent, tokenize = True):
    """
    Gets all the possible combinaisons of words in a suffled sentence of n words

    Arguments:
        sent -- string, the sentence we want to analyze
        tokenize -- bool (default=True) add a tokenization step to the sentence
    
    Returns:
        sentVars -- numpy array of shape (n!, n), containing all the possible variations of said sentence
    """
    if (tokenize):
        sent = TextCleanup(sent)
    sent = sent.strip().split()
    sentVars = permutations(sent)

    return sentVars

def OneStepPrediction(sent, words, model):
    """
    Performs one prediction step. Takes the input incomplete sentence, and predicts the next most probable word not previously drawn

    Arguments:
        sent -- str, the input sentence
        words -- the list of remaining words to be sorted
        model -- the KenLM model instance

    Returns: 
        newSent -- str, the completed sentence
        words -- the remaining words to be assigned
    """ 

    perplexity = model.perplexity(sent + ' ' + words[0])

    for word in words:
        sentCandidate = sent + ' ' + word
        perpCandidate = model.perplexity(sentCandidate)
        if perpCandidate <= perplexity:
            perplexity = perpCandidate
            newSent = sentCandidate
            chosenWord = word
    words.remove(chosenWord)

    return newSent, words

def SortSentenceV1(sent, ngram_model):
    """
    Get the word combination with the minimum of perplexity from all possible permutations of words in a sentence.
    
    Arguments:
        sent -- string, the shuffled sentence
        ngram_model -- the n-gram model to be used
    
    Returns:
        sorted_sent -- the sorted sentence
        perplexity -- float, the perplexity value of said sentence
    """
    perplexity = 10000000
    sorted_sent = ''
    sentVars = SentVariations(sent)
    for perm in sentVars:
        perp = ngram_model.perplexity(' '.join(perm))
        if perp < perplexity:
            perplexity = perp
            sorted_sent = perm
    
    return sorted_sent, perplexity
        
def SortSentenceV2 (words, model):
    """
    Predict a sentence from a list of words, using a previsouly trained KenLM n-gram model

    Arguments:
        words -- the list of words to place in order  
        model -- the KenLM n-gram model  
    
    Returns:
        outSent -- the output sentence  
        perplexity -- the perplexity of the output sentence
    """
    sents = []
    
    # Every word can be the starting word
    for word in words:        
        remainingWords = words.copy()
        remainingWords.remove(word)
        sent = word
        
        # While there is still words to be sorted, perform one prediction step
        while len(remainingWords) > 0:
            sent, remainingWords = OneStepPrediction(sent, remainingWords, model)
        sents.append(sent)
        

    # Get the sentence with the minimum of perplexity from the list of sentences
    outSent, perplexity = GetLeastPerplexSent(sents, model)
    
    return outSent, perplexity, sents

def SortSentenceV3 (words, model):
    """
    Predict a sentence from a list of words, using a previsouly trained KenLM n-gram model

    Arguments:
        words -- the list of words to place in order  
        model -- the KenLM n-gram model  
    
    Returns:
        outSent -- the output sentence  
        perplexity -- the perplexity of the output sentence
    """
    sents = []
    
    # Every word can be the starting word
    for ngram in GetNgrams(words, model):        
        ngram = ngram.split()
        remainingWords = words.copy()
        # If the ngram ends with '.', pop the end
        if ngram[-1] == '.':
            ngram.pop()
        for word in ngram:
            remainingWords.remove(word)
        sent = ' '.join(ngram)
        
        # While there is still words to be sorted, perform one prediction step
        while len(remainingWords) > 0:
            sent, remainingWords = OneStepPrediction(sent, remainingWords, model)
        sents.append(sent)
        

    # Get the sentence with the minimum of perplexity from the list of sentences
    outSent, perplexity = GetLeastPerplexSent(sents, model)
    
    return outSent, perplexity, sents

def SortSentenceSeq2Seq(input_seq, encoder_model, decoder_model, word_to_index, index_to_word, maxLen):
    """
    Sorts an input sentence using a preivously trained seq2seq architecture

    Arguments:
        input_seq -- str, the input sentence
        encoder_model -- the encoder model from the seq2seq architecture
        decoder_model -- the decoder model from the seq2seq architecture
        word_to_index -- dict mapping every word to its corresponding index
        index_to_word -- dict mapping every index to its corresponding word
        maxLen -- the maximum length of the output string
    
    Returns:
        the decoded sentenced as a string
    """
    vocabSize = len(word_to_index.keys())
    # Encode the input as state vectors.
    input_seq = input_seq.reshape(1,maxLen).astype(np.int32)
    states_value = encoder_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = np.array(word_to_index['<bos>']).reshape(1,1)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, states_value = decoder_model.predict([target_seq,  states_value])
        # Sample a token
        sampled_word_seq_index = np.argmax(output_tokens[:,:,input_seq.astype(np.int32)[0]])
        sampled_word_index = input_seq[0,sampled_word_seq_index]
        sampled_word = index_to_word[sampled_word_index]
        decoded_sentence.append(sampled_word)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '<eos>' or len(decoded_sentence) > maxLen):
            stop_condition = True

        # Update the target sequence (of length 1).        
        target_seq = np.array([[sampled_word_index]])

        # Update states
        states_value = states_value

    return ' '.join(decoded_sentence)

def BeamSearch(inputs_probas, k):
    """
    Uses a beamsearch algorithm to get the k most likely sequences, given input probabilities
​
    Arguments:
        inputs_probas -- Probabilities of the inputs
        k -- Number of sequences we want to keep
​
    Returns:
        sequences -- The k most likely sequences
    """
    sequences = [[[], 0.0]]
    for input in inputs_probas:
        all_candidates = []
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(input)):
                candidate = [seq + [j], score - np.log(input[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]
    return sequences