# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 01:56:22 2021

@author: Juand
"""
import urllib


'''
This function take the devoir3-train.txt URL and outputs a dict that contains as key the misspelled word
and as value the correct word.
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

'''
This function opens the output of the previous program (devoir3-sortie.txt).
For each line, it checks if the misspelled word corresponds to a misspelled word
in the training data. If it does, then it checks whether the expected correction 
of the training data is present. 
This function provides a hard test and a soft test. 

1.Hard test:

    line[0] = misspelled word
    line[1] = 1st proposed correction
    line[2] = 2nd proposed correction
    line[3] = 3rd proposed correction
    line[5] = 4th proposed correction
    line[6] = 5th proposed correction

    If the actual correction corresponds to line[1], then "the_sum2" gains 5 points.
    if the actual correction corresponds to line[2], then "the_sum2" gains 4 points.
    and so on...

    Then, the_sum2 is divided by the number of misspelled words*5, and multiplied by 100 to 
    get a percentage of succesful correction. 

    In an ideal world, every expected correction would correspond to the 1st proposed correction,
    granting 5 points per misspelled world, and this would yield a percentage of 100%.

2. Soft test:

    This simply checks if the correct word is present in the list of possible correct words.

    If it does, "the_sum" gauns 1 point.
    The_sum is divided by the number of misspelled words and multiplied by 100 to get a
    percentage of succesful correction. 

    In an ideal world, every expected correction would be present in the list of possible 
    corrections yielding a grade of 100%.
'''  

def eval():
    data_train = toDictTrain()
    data_train = data_train.items()
    #print(data_train)

    file1 = open('devoir3-sortie.txt', 'r')
    Lines = file1.readlines()
    the_sum=0
    the_sum2=0
    
    #Soft test
    for line in Lines:
        clearLine=False
        line=str(line)
        line=line.split()

        for pair in data_train:
            if clearLine==True:
                break
            
            j=0
            for word in line:  
                if j!=0:
                    if word == pair[1]:

                        the_sum+=1
                        clearLine=True
                        break
   
                j+=1
    
    #Hard test
    for line in Lines:
        clearLine=False
        line=str(line)
        line=line.split()
        for pair in data_train:
            expected_correction=pair[1]
            if line[0]==pair[0]:
                
                if len(line)>=2 and line[1]==expected_correction: 
                    the_sum2+=5
                elif len(line)>=3 and line[2]==expected_correction:
                    the_sum2+=4
                elif len(line)>=4 and line[3]==expected_correction:
                    the_sum2+=3
                elif len(line)>=5 and line[4]==expected_correction:
                    the_sum2+=2
                elif len(line)>=6 and line[5]==expected_correction:
                    the_sum2+=1
                break
       

    print('soft test:')
    print((the_sum/len(Lines))*100, "%")
    print('hard test:')
    print((the_sum2/len(Lines))*20, "%")


eval()
