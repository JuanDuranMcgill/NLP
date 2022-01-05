# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:53:12 2021

@author: Juand
"""

import numpy as np
import matplotlib.pyplot as plt


#The only purpose of this code is to plot the curves.
def loader(file):
    content = []
    with open(file, "r") as output:
        for line in output:
            content.append(float(line))
    #print(content)
    return content
    
    
temps = loader("temps.txt")

types = loader("types.txt")

#print(len(types))

tranches = np.arange(1,100)

#print(tranches)
plt.figure(0)
plt.plot(tranches,temps,'g')
plt.title("Temps pour compter mots en fonction des tranches")
plt.xlabel("Tranches")
plt.ylabel("Temps")
plt.savefig("temps.png")

plt.figure(1)
plt.plot(tranches,types,'g ')
plt.title("Types de mots en fonction des tranches")
plt.xlabel("Tranches")
plt.ylabel("Types")
plt.savefig("Types.png")