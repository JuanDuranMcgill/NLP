"""
This program is doing a series of tests to evaluate the impact of hyperparameters on the gensim Word2Vec model

Created on Wed Oct 13 15:47
@authors: Juan Felipe Duran, Maxime Monrat
"""

# -*- coding: utf-8 -*-

import gensim, logging
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences=[]
##############################################################################
         #Chargement de 5 tranches du 1BWC dans la liste "sentences" (1/6)
##############################################################################

n=1

for i in range(5): 
           
    if n!=10:
        string =r"C:\Users\Juand\Desktop\UdeM\IFT6285\dev1\training-monolingual.tokenized.shuffled\news.en-0000{num}-of-00100".format(num=n)
    else:
        string =r"C:\Users\Juand\Desktop\UdeM\IFT6285\dev1\training-monolingual.tokenized.shuffled\news.en-000{num}-of-00100".format(num=n)
    f = open(string,"r",encoding="utf8")
    
         
    Lines = f.readlines()
    
    for line in Lines:
        
        wordList = line.split()
        sentences.append(wordList)
    
    
    f.close()
 
##############################################################################
        #Valeur a utiliser pour faire l'analyse du modele (2/6)
##############################################################################

neg=[0,5,10,15,20,15] #Exemples negatifs
win=[2,4,6,8,10,12,14,16] #Taille de contexte
vecs=[200,250,300,350,400,450,500,550,600] #Taille de vecteur
 


##############################################################################
                    #Analyse pour exemples negatifs (3/6)
##############################################################################

negacc=[]
negtime=[]

for x in neg:
    ti = time.time()
    model = gensim.models.Word2Vec(sentences, min_count=1,negative=x)
    tf = time.time()
    
    t =tf-ti
    
    model = model.wv
    
    evalu=model.evaluate_word_analogies(r'C:\Users\Juand\Desktop\UdeM\IFT6285\dev4\questions-words.txt')
    
    
    negacc.append(evalu[0])
    negtime.append(t)
    
##############################################################################
                    #Analyse pour taille du contexte (4/6)
##############################################################################

winacc=[]
wintime=[]
    
for x in win:
    ti = time.time()
    model = gensim.models.Word2Vec(sentences, min_count=1,window=x)
    tf = time.time()
    
    t =tf-ti
    
    model = model.wv
    
    evalu=model.evaluate_word_analogies(r'C:\Users\Juand\Desktop\UdeM\IFT6285\dev4\questions-words.txt')
    
    
    winacc.append(evalu[0])
    wintime.append(t)
    
    
##############################################################################
                    #Analyse pour taille de vecteur (5/6)
##############################################################################

vecacc=[]
vectime=[]
    

for x in vecs:
    ti = time.time()
    model = gensim.models.Word2Vec(sentences, min_count=1,vector_size=x)
    tf = time.time()
    
    t=tf-ti
    
    model = model.wv
    
    evalu=model.evaluate_word_analogies(r'C:\Users\Juand\Desktop\UdeM\IFT6285\dev4\questions-words.txt')
    
    
    vecacc.append(evalu[0])
    vectime.append(t)


##############################################################################
                    #Sauvegarde des valeurs dans un "text file" (6/6)
##############################################################################

f = open("dev4_p2_output.txt","w")
f.write("#####Negative_sampling:")
f.write("\n")
f.write("Value:")
f.write("\n")
f.write(str(neg))
f.write("\n")
f.write("Accuracy:")
f.write("\n")
f.write(str(negacc))
f.write("\n")
f.write("Time")
f.write("\n")
f.write(str(negtime))
f.write("\n")
f.write("#####Window:")
f.write("\n")
f.write("Value:")
f.write("\n")
f.write(str(win))
f.write("\n")
f.write("Accuracy:")
f.write("\n")
f.write(str(winacc))
f.write("\n")
f.write("Time")
f.write("\n")
f.write(str(wintime))
f.write("\n")
f.write("#####Vector_Size:")
f.write("\n")
f.write("Value:")
f.write("\n")
f.write(str(vecs))
f.write("\n")
f.write("Accuracy:")
f.write("\n")
f.write(str(vecacc))
f.write("\n")
f.write("Time")
f.write("\n")
f.write(str(vectime))

    
f.close()