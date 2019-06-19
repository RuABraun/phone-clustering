import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
import random

def main(infa, infb):
    data = np.load(infa)
    dataaug = np.load(infb)

    fig = plt.figure()

    idx = random.randint(0, len(data)-1)
    fig.add_subplot(4, 1, 1)
    plt.imshow(data[idx].T)
    fig.add_subplot(4, 1, 2)
    plt.imshow(dataaug[idx].T)

    idx = random.randint(0, len(data)-1)
    fig.add_subplot(4, 1, 3)
    plt.imshow(data[idx].T)
    fig.add_subplot(4, 1, 4)
    plt.imshow(dataaug[idx].T)

    plt.show()


import plac; plac.call(main)
