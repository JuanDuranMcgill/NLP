import numpy as np
import matplotlib.pyplot as plt

def plotter(i):
    time = []
    size = []
    perplexity = []

    j = 1
    for x in range(i):
        f = open("t{je}_info".format(je=j))
        Lines = f.readlines()
        time.append(float(Lines[0]))
        size.append(float(Lines[1])/100000000)
        perplexity.append(float(Lines[2]))
        j+=1


    tranches = list(range(1, i+1))
    '''
    plt.plot(tranches, time)
    plt.xlabel("tranches")
    plt.ylabel("temps(s)")
    plt.title("temps en fonction de tranches")
    plt.savefig("temps_tranches.png")
    '''
    plt.plot(tranches, perplexity)
    plt.xlabel("tranches")
    plt.ylabel("perplexity")
    plt.title("perplexity en fonction de tranches")
    plt.savefig("perplexity_tranches.png")

plotter(87)




