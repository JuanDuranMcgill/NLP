"""
Program used to get the results in the assignment 7 of the class IFT6285
University of Montreal, Autumn 2021

@authors: Maxime Monrat, Juan Felipe Duran, 2021, Nov 17
"""
from nltk.corpus import treebank
from nltk.sem.evaluate import Model
from nltk.tag import CRFTagger, RegexpTagger, BrillTaggerTrainer
from nltk.tag.brill import Pos, Word, brill24, nltkdemo18, nltkdemo18plus, fntbl37
from nltk.tbl.template import Template
from nltk import pos_tag, pos_tag_sents, word_tokenize
import numpy as np
import pandas as pd
import os, re, string, unicodedata
import pickle

from nltk.tokenize.casual import REGEXPS
import util

TRAIN_DATA = treebank.tagged_sents()[:3000]
TEST_DATA = treebank.tagged_sents()[3000:]
TEST_SENTS = treebank.sents()[3000:]
REGEXPTAG = RegexpTagger([
(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
(r'(The|the|A|a|An|an)$', 'AT'),   # articles
(r'.*able$', 'JJ'),                # adjectives
(r'.*ness$', 'NN'),                # nouns formed from adjectives
(r'.*ly$', 'RB'),                  # adverbs
(r'.*s$', 'NNS'),                  # plural nouns
(r'.*ing$', 'VBG'),                # gerunds
(r'.*ed$', 'VBD'),                 # past tense verbs
(r'.*', 'NN')                      # nouns (default)
])


def TrainCRF():
    model = CRFTagger(feature_func=util.word2features)
    model._verbose=True
    model.train(TRAIN_DATA, 'model3.crf.tagger')
    model.tag_sents(TEST_SENTS)
    model.evaluate(TEST_DATA)

def TestCRF(modelFile):
    model = CRFTagger(feature_func=util.word2features)
    model.set_model_file(modelFile)
    model.tag_sents(TEST_SENTS)
    print(model.evaluate(TEST_DATA))

def TrainTBL():

    # base = REGEXPTAG
    base = CRFTagger(feature_func=util.CustomFeatureFunc)
    base.set_model_file('model3.crf.tagger')

    # Set up templates
    Template._cleartemplates() #clear any templates created in earlier tests
    templates = brill24()
    tt = BrillTaggerTrainer(base, templates, trace=3)
    tagger= tt.train(TRAIN_DATA, max_rules=100)
    print('Accuracy score: ' + str(tagger.evaluate(TEST_DATA)))
    print("Learned Rules: " + str(tagger.rules()[:20]))
 
    with open('brill-regexbased.pkl', 'wb') as fout:
        pickle.dump(tagger, fout)

    # with open(r'brill-regexbased.pkl', "rb") as fin:
    #     tagger = pickle.load(fin)

    print('Accuracy score: ' + str(tagger.evaluate(TEST_DATA)))
    print("Learned Rules: " + str(tagger.rules()[:20]))

def TBLtests():
    perfs = np.zeros((1,10))
    templates = brill24()
    # base1 = REGEXPTAG
    with open(r'brill-regexbased.pkl', "rb") as fin:
        tagger = pickle.load(fin)
    try:
        for i in range(perfs.shape[1]):
            base = tagger
            Template._cleartemplates()
            templates = brill24()
            tt = BrillTaggerTrainer(base, templates, trace=3)
            tagger= tt.train(TRAIN_DATA, max_rules=100)
            perfs[0,i] = tagger.evaluate(TEST_DATA)
    except KeyboardInterrupt:
        np.savetxt('BrillRecursive.txt', perfs, delimiter='\t')
    np.savetxt('BrillRecursive.txt', perfs, delimiter='\t')
        


usrInput = ''
while(usrInput != 'exit'):
    usrInput = input("What do you want to do?\n\t1 -- Train a CRF tagger\n\t2 -- Train a TBL tagger\n\t3 -- Evaluate an existing model\n\texit -- exit the program safely\n")
    
    if(usrInput == '1'):    
        print("C'est parti mon kiki")
        TrainCRF()
    
    elif(usrInput == '2'):
        print("En voiture Simone")
        # TrainTBL()
        TBLtests()
    
    elif(usrInput == '3'):    
        print('Models in current working directory:')
        files = util.Get_Models_Names(os.getcwd())
        n = 1
        for f in files:
            print('\t' + str(n) + ' -- ' + f)
            n += 1
        usrInput_eval = int(input('Which model do you want to evaluate?\n'))
        if((files[usrInput_eval-1]).split('.')[-2] == 'crf'):
            TestCRF(files[usrInput_eval-1])

    elif(usrInput == 'exit') :
        print("On se reverra Joe, on se reverra.")
    
    else:
        print("Commence pas à raconter n'importe quoi mon petit pèpère")