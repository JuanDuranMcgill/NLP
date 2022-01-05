# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 01:45:32 2021

@author: Juand
"""
##############################################################################
########################(1/4) Imports and load functions######################
##############################################################################

import urllib
import jaro
import operator
import time

'''
This function take the voc-1bc.txt URL and outputs a dict that contains as key the word
as as value the number of times it appears in the corpus.
'''
def toDict():  
    url = "http://www-labs.iro.umontreal.ca/~felipe/IFT6285-Automne2020/voc-1bwc.txt"
    file = urllib.request.urlopen(url)   
    a=[]
    for line in file:
    	decoded_line = line.decode("utf-8")
    	a.append(decoded_line)
     
    keys=[]
    values=[]
    for e in a: 
        pair=e.split()
        values.append(float(pair[0]))
        keys.append(pair[1])
    
    WORDS=dict(zip(keys,values))
    return WORDS

WORDS =toDict()

'''
This function take the devoir3-train.txt URL and outputs a dict that contains as 
key the misspelled word and as value the correct word.
'''
def toDictTrain():  
    url = "http://www-labs.iro.umontreal.ca/~felipe/IFT6285-Automne2020/devoir3-train.txt"
    file = urllib.request.urlopen(url)   
    a=[]
    for line in file:
        decoded_line = line.decode("utf-8")
        a.append(decoded_line)
     
    keys=[]
    values=[]
    for e in a: 
        pair=e.split()
        values.append(pair[0])
        keys.append(pair[1])
    
    WORDS=dict(zip(values,keys))
    return WORDS

##############################################################################
########################(2/4) Helper functions################################
##############################################################################
'''
This function takes as input a word and outputs all possible combinations or strings
that are 1 edit away.
'''
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


'''
This function takes as input a word and outputs all possible combinations or strings
that are 2 edits away.
'''
def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

'''
This function takes as input a word and outputs all possible combinations or strings
that are 3 edits away.
'''
def edits3(word): 
    "All edits that are two edits away from `word`."
    return (e3 for e1 in edits1(word) for e2 in edits1(e1) for e3 in edits1(e2))

'''
This function takes as input a set of strings and returns the set of strings that
are actually known words from the corpus
'''
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

'''
This function takes as input a word and outputs the empirical probability that
it appears in the corpus by taking #times it appears divided by all words in corpus

'''
def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

'''
This function takes as input a list of words and a list of numbers, which correspond 
to the probability each word appears in the corpus. 
This function outputs a list of words sorted by ther probability
'''
def listSorter(words,num):
    
    dic = dict(zip(words,num))
    dic_sorted = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
    dic_sorted=dict(dic_sorted)
    
    output=[]
    for word in dic_sorted:
        output.append(word)
    return output

'''
This function transforms a word into a vec.
'''
def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

'''
This function calculates the cosinus distance
between two vectors.
'''
def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]

##############################################################################
########################(3/4) Distance functions##############################
##############################################################################
'''
This function takes as input a word. It then looks for all possible combinations of 
characters that are 1 or 2 edits away. Then it selects, from those combinations, the words
that are actually present in the corpus voc-1bwc.txt. Then, it assigns to each word its corresponding
probability of appearing in the corpus and rearranges them from most probable to least probable. 
Finally, it takes the 5 first words in the list and selects them as the possible correction.
The output is the 5 most probable words that are 1 or 2 edits away from the initial (mistaken) word.
'''
def edition(word): 
    WORDS = toDict()
    output=[]
    d=list(known(edits1(word).union(edits2(word))))
    d_P=[]
    for e in d:
        d_P.append(P(e))       
    output = listSorter(d,d_P)
    return output[0:5]

'''
This function implements the jaro-winkler distance.
It takes as input a word and the list of "known words" from the provided corpus.
It outputs a list of the five words with the highest jaroc-winkler score.
'''
    
def jaroc(word,kw):
    metric=[]
    for x in kw:
        metric.append(jaro.jaro_winkler_metric(x,word))    
    output=listSorter(kw,metric) 
    return output[0:5]

'''
This function implements the cosinus distance
It takes as input a word and the list of "known words" from the provided corpus.
It outputs a list of the five words with the highest cosinus distance score.
'''
def cosinus(word,kw):
    word = word2vec(word)
    metric=[]
    for x in kw:
        x = word2vec(x)
        metric.append(cosdis(x,word))    
    output=listSorter(kw,metric) 
    return output[0:5]

'''
This function takes as input the training dataset. 
It takes the first column of the document "devoir3-train.txt" 
(which are the misspelled words) and for each word, it computes
the five most probable corrections. Then, it prints in the file "devoir3-sortie.txt"
a misspelled word and its corresponding corrections per line. 
This function can either run the cosinus distance, the jaroc distance, or the edition distance. 
''' 
def corrige(url,method):

    file = urllib.request.urlopen(url)   
    mw=[] #short for misspelled words
    for line in file:
        first_column=[]
        decoded_line = line.decode("utf-8")
        first_column=decoded_line.split()
        #mw.append(str(decoded_line.strip()))
        mw.append(str(first_column[0]))
    
    f = open("devoir3-sortie.txt", "w")  
    length=len(mw)
    j=1
    for w in mw:
        line=w
        if method == "jaro":
            kw=[] #short for known words
            for x in WORDS:
                kw.append(str(x))
            corrections = jaroc(w,kw)
        if method == "edition":
            corrections = edition(w)
        if method == "cosinus":
            kw=[] #short for known words
            for x in WORDS:
                kw.append(str(x))
            corrections = cosinus(w,kw)
        for e in corrections:
            line+="\t"
            line+=e
        
        f.write(line)
        f.write("\n")
        print(j,"/",length)
        j+=1
    f.close()
    
##############################################################################
########################(4/4) Function calls##################################
##############################################################################    
    
url = "http://www-labs.iro.umontreal.ca/~felipe/IFT6285-Automne2020/devoir3-train.txt"
t1=time.time()
corrige(url,"edition")
t2=time.time()

print("done")
print("time: ", t2-t1)