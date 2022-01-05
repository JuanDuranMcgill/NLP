
#Calcul du score en fonction de phrases considerees

import os 

a = "sacrebleu out2.txt -i out1.txt -m bleu -b -w 4 >> scorePer10SentIncrement.txt"

temp1=[]
temp2=[]





for i in range(3000):
    if i==0:
        continue
    if i%10==0:
        temp1=[]
        temp2=[]
        f = open("out.txt", "r")
        g = open("ref.txt","r")
        for e in range(i):
            temp1.append(f.readline())
            temp2.append(g.readline())
        f.close()
        g.close()

        b = open("out1.txt","w")
        for t in temp1:
            b.write(t)
        b.close()
        c = open("out2.txt","w")
        for t in temp2:
           c.write(t)

        c.close()
        os.system(a)

#Calcul du score en fonction de la longueur de phrases
'''
a = "sacrebleu refs.txt -i outs.txt -m bleu -b -w 4 >> scoreSentenceLength.txt"

temp1=[]
temp2=[]



for e in range(1,70):
    f=open("out.txt","r")
    g=open("ref.txt","r")
    lines1=f.readlines()
    lines2=g.readlines()
    out=[]
    ref=[]
    for l1,l2 in zip(lines1,lines2):
        if len(l1.split())<=e:
            out.append(l1)
            ref.append(l2)
    f.close()
    g.close()
    b=open("outs.txt","w")
    c=open("refs.txt","w")
    for o,r in zip(out,ref):
        b.write(o)
        c.write(r)
    b.close()
    c.close()
    os.system(a)

'''


