# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:01:07 2021

@author: Juand
"""

import nltk 
import spacy

from nltk.corpus import treebank
from nltk.tag import CRFTagger
from nltk.tag.api import TaggerI
from nltk.tag import BrillTagger

from nltk.tag import brill, brill_trainer
  
def train_brill_tagger(initial_tagger, train_sents, **kwargs):
    templates = [
            brill.Template(brill.Pos([-1])),
            brill.Template(brill.Pos([1])),
            brill.Template(brill.Pos([-2])),
            brill.Template(brill.Pos([2])),
            brill.Template(brill.Pos([-2, -1])),
            brill.Template(brill.Pos([1, 2])),
            brill.Template(brill.Pos([-3, -2, -1])),
            brill.Template(brill.Pos([1, 2, 3])),
            brill.Template(brill.Pos([-1]), brill.Pos([1])),
            brill.Template(brill.Word([-1])),
            brill.Template(brill.Word([1])),
            brill.Template(brill.Word([-2])),
            brill.Template(brill.Word([2])),
            brill.Template(brill.Word([-2, -1])),
            brill.Template(brill.Word([1, 2])),
            brill.Template(brill.Word([-3, -2, -1])),
            brill.Template(brill.Word([1, 2, 3])),
            brill.Template(brill.Word([-1]), brill.Word([1])),
            ]
      
    # Using BrillTaggerTrainer to train 
    trainer = brill_trainer.BrillTaggerTrainer(
            initial_tagger, templates, deterministic = True)
      
    return trainer.train(train_sents, **kwargs)
'''
import spacy
model = "en_core_web_sm" # try also the _lg one
nlp = spacy.load(model,
disable=["parser", "ner"]) # to go faster
# we want to do this:
# doc = nlp(’hello world !’)
#
# but the tokenization would change from the one in treebank
# which would cause problems with the function evaluate
# so instead do this more convoluted thing:
tokens_of_my_sentence = ['hello', 'world', '!']
doc = spacy.tokens.doc.Doc(nlp.vocab, words=tokens_of_my_sentence)
for _, proc in nlp.pipeline:
    doc = proc(doc)
# now doc is ready:
for t in doc:   
    print(f'{t.text:20s} {t.tag_}')

'''



train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]



model = "en_core_web_sm" # try also the _lg one
model2 = "en_core_web_lg"


mod=[model,model2]

mad=["sm","lg"]

for m,n in zip(mod,mad):
    
    import spacy
    nlp = spacy.load(m,
    disable=["parser", "ner"])
    
    
    class spacyTagger(TaggerI):
        def tag(self, tokens):
         doc = spacy.tokens.doc.Doc(nlp.vocab, words=tokens)
         for _, proc in nlp.pipeline:
             doc = proc(doc)
         res = []
         for i in doc:
             res.append((str(i), i.tag_))
         return res
     
    tagger=spacyTagger()
    

    brill_tag = train_brill_tagger(tagger, train_data)
    print("training for ",n," done.")
    
    
    '''
    toeval=[]
    i=0
    for e in treebank.tagged_sents():
        if i<3000:
            i+=1
            continue
        print(i)
        toeval.append(e)
        i+=1
        #if i ==len(treebank.tagged_sents()):
        if i ==3010:
            break
        
    '''    
        #print(len(toeval))
        #print(tagger.evaluate(toeval))
    
    with open('dev7_Brill_performance_modele_{a}.txt'.format(a=n), 'w') as f:
        f.write(str(brill_tag.evaluate(test_data)))


