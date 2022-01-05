
import kenlm
import numpy as np

m = kenlm.LanguageModel('t.arpa')

array=[]

# Using readlines()
f = open('testReal', 'r')
Lines = f.readlines()
 

for line in Lines:
        array.append(m.perplexity(line))



print("average:",np.mean(array))
print("max:",np.max(array))
print("min:",np.min(array))







