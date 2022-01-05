# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:51:21 2021

@author: Juand
sources:
https://www.geeksforgeeks.org/python-count-occurrences-of-each-word-in-given-text-file-using-dictionary/
https://stackoverflow.com/questions/13351981/compare-strings-based-on-alphabetical-
https://www.kite.com/python/answers/how-to-sort-a-dictionary-by-key-in-python
"""
import numpy as np
import time


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def isAlphanumeric(inputString):
    return inputString.isalnum()

# Create an empty dictionary
def wordCounter(i):
    
    if i>99:
        print("input can only be max 99")
        return
    
    array = np.arange(1,i+1)
    
    #print(array)
    d = dict()
    for x in array:
        #text = open("000{fname}".format(fname = x),"r")
        
        if x<10:
            text = open("training-monolingual.tokenized.shuffled/news.en-0000{n}-of-00100".format(n=x), "r",encoding="utf8")
        else:
            text = open("training-monolingual.tokenized.shuffled/news.en-000{n}-of-00100".format(n=x), "r",encoding="utf8")
     
        # Loop through each line of the file  
        for line in text:
            # Remove the leading spaces and newline character
            line = line.strip()
          
            # Convert the characters in line to 
            # lowercase to avoid case mismatch
            line = line.lower()
          
            # Split the line into words
            words = line.split(" ")
           
            # Iterate over each word in line
            for word in words:
                
                
                if isAlphanumeric(word)==False:
                    word = "__SYM__"
                if has_numbers(word)==True:
                    word="__NUM__"
                # Check if the word is already in dictionary
                if word in d:
                    # Increment count of word by 1
                    d[word] = d[word] + 1
                else:
                    # Add the word to dictionary with count 1
                    d[word] = 1
      
      
    #sort by key
    d2 = {k: v for k,v in sorted(d.items())}
    
    #freq_top1000(d)
    #freq_less1000(d)
    
    # Print the contents of dictionary
    countInstance = 0
    countClass = 0
    for key in list(d2.keys()):
        #print(key, ":", d[key])
        countInstance+=d2[key]
        countClass+=1
              
    #print("there are ", countInstance, " words and ", countClass," types.")
    
    print("types: ", countClass)
    print("words: ", countInstance)
    return d2,countClass,countInstance
    
def countStuff():
    #This function was only used to get the datapoints for the curve figures. 
    temps=[]
    types=[]
    for x in np.arange(1,100):
        start_time = time.time()  
        a,b,c = wordCounter(x)
        times = time.time()-start_time
        temps.append(times)
        types.append(b)
        print ("My program ", x, " took", times, "to run")
    
    
    with open("temps.txt", "a") as output:
        for x in temps:
            output.write(str(x)+"\n")
    with open("types.txt", "a") as output:
        for x in types:
            output.write(str(x)+"\n")
        
def freq_top1000(d):
    d3={k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    with open("freq-top1000.txt", "a",encoding="utf-8") as output:
        i=0
        for key, value in reversed(d3.items()):
            key = str(key)
            value = str(value)
            line = key
            output.write(line+" ")
            i+=1
            if i==1000:
                break
def freq_less1000(d):
    d3={k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    with open("freq-less1000.txt", "a",encoding="utf-8") as output:
        i=0
        for key, value in d3.items():
            key = str(key)
            value = str(value)
            line = key
            output.write(line+" ")
            i+=1
            if i==1000:
                break

start_time = time.time()  
wordCounter(99)
print ("My program took", time.time() - start_time, "to run")