# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 04:21:01 2021

@author: Juand
"""

import spacy
import spacy.lang.en
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#nlp = spacy.load('en_core_web_sm')
#nlp2=spacy.load('en_core_web_md')
#nlp3=spacy.load("en_core_web_lg")



r'''             (dev6_q4) Extraction d'informations interessantes
df1 = pd.read_csv('triplet_modele_q4_sm.csv')


w='president'
with open('{a}.csv'.format(a=w),'w',newline="") as f:
    writer = csv.writer(f)
    for s,v,o in zip(df1['sujet'],df1['verbe'],df1['object']):
        if str(s).lower()==w:
            if not pd.isna(s) and not pd.isna(v) and not pd.isna(o):
                writer.writerow([s,v,o])
                print(s,v,o)
f.close()

'''











r'''          (dev6_q3) Visualisation des triplets en fonction des phrases
mod="lg"
df = pd.read_csv("tripletsPerSentence/tripletsPerSentence_{ne}.csv".format(ne=mod))

se = df["sentences"].to_list()

te = df["triplets"].to_list()

print(te)
print(se)

plt.ylabel("triplets")
plt.xlabel("sentences")
plt.title("triplets-as-a-function-of-sentences-model-{ne}".format(ne=mod))
plt.plot(se,te,'.',color="r")
plt.savefig("plot-{ne}".format(ne=mod))
'''



r'''          (dev6_q3)   Function for extraction of triplets. 
sentences=[]
with open(r"C:\Users\Juand\Desktop\UdeM\IFT6285\dev1\training-monolingual.tokenized.shuffled\news.en-00001-of-00100",encoding="utf-8") as fp:
    for line in fp:
        sentences.append(line)
        
sentences=sentences[0:200000]





def tripPerSent(mod):

    df1 = pd.read_csv (r'triplet_modele_{ne}.csv'.format(ne=mod))
    
    
    info=[]
    for s,v,o in zip(df1["sujet"],df1["verbe"],df1["object"]):
        if pd.isna(s) or pd.isna(v) or pd.isna(o):
            info.append(1)
        else:
            info.append(0)
            
    print(len(info))       
    triplets=[0]
    sen=[0,5e3,10e3,15e3,20e3,25e3,30e3,35e3,40e3,45e3,50e3]       
    for s in sen:
        if s==0:
            continue
        counter=0
        for n,i in zip(info,range(len(info))):
            if n==0:
                counter+=1
            if i+1==s:
                triplets.append(counter)
                break
            
    sen=pd.Series(sen)
    tri=pd.Series(triplets)
    
    
    df=pd.concat({"sentences":sen,"triplets":tri},axis=1)
    #df=pd.DataFrame({"sentences":sen,"triplets":tri})
    

    
    
    df.to_csv('tripletsPerSentence_{ne}.csv'.format(ne=mod),index=False)
    print("done with ",mod)




def triplets(sent,model,n):
    suj1=[]
    verb1=[]
    obj1=[]
    
    i=0
    
    for s in sentences:
        an=model(s)

        
        s=False
        v=False
        o=False
        
        for tok in an:
            if str(tok.dep_)=="nsubj" and s==False:
                if str(tok.pos_)=="PROPN":
                    suj1.append("PROPN")
                else:
                    suj1.append(tok.text)
                s=True
                continue
                
            if str(tok.pos_)=="VERB" and v==False:
                verb1.append(tok.text)
                v=True
                continue
            if str(tok.dep_)=="dobj" and o==False:
                obj1.append(tok.text)
                o=True
                continue
            if s==True and v==True and o==True:
                break

        if s==False:
            suj1.append(np.nan)
        if v==False:
            verb1.append(np.nan)
        if o==False:
            obj1.append(np.nan)
        i+=1
        if i%1000==0:
            print(n+" ",i,"/",len(sentences))
        
                
    
    suj1=pd.Series(suj1)
    verb1=pd.Series(verb1)
    obj1=pd.Series(obj1)
    
    dic1={"sujet":suj1,"verbe":verb1,"object":obj1}
    
    df1 = pd.concat(dic1,axis=1)
    
    df1.to_csv('triplet_modele_q4_{ne}.csv'.format(ne=n),index=False)
    

triplets(sentences,nlp,"sm")
#triplets(sentences,nlp2,"md")
#triplets(sentences,nlp3,"lg")
    
#tripPerSent("sm")
#tripPerSent("md")
#tripPerSent("lg")

'''





'''    (dev6_q1) Analyse des phrases en fonction  du temps pour trois modeles

sents=[0,2e4,4e4,6e4,8e4,10e4,12e4,14e4,16e4,18e4,20e4]
time1=[0]
time2=[0]
time3=[0]

print("go")
i=1
for e in sents:
    if e==0:
        continue
    t1=t.time()
    for s,count in zip(sentences,range(len(sentences))):
        tmp=nlp(str(s))
        if count==e:
            break
    t2=t.time()
    t3=t.time()
    for s,count in zip(sentences,range(len(sentences))):
        tmp=nlp2(str(s))
        if count==e:
            break
    t4=t.time()
    t5=t.time()
    for s,count in zip(sentences,range(len(sentences))):
        tmp=nlp3(str(s))
        if count==e:
            break
    t6=t.time()
    
    time1.append(t2-t1)
    time2.append(t4-t3)
    time3.append(t6-t5)
    print(i,"/",len(sents)-1," done")
    i+=1
    
sents=pd.Series(sents)
time1=pd.Series(time1)
time2=pd.Series(time2)
time3=pd.Series(time3)

dic1={"sentences":sents,"time":time1}
dic2={"sentences":sents,"time":time2}
dic3={"sentences":sents,"time":time3}

df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
df3 = pd.DataFrame(dic3)

df1.to_csv('temps-danalyse-en_core_web_sm.csv',index=False)
df2.to_csv('temps-danalyse-en_core_web_md.csv',index=False)
df3.to_csv('temps-danalyse-en_core_web_lg.csv',index=False)
'''