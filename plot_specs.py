import matplotlib.pyplot as plt
import numpy as np
import random

plt.style.use('ggplot')


def main(infa, infb, infc):
    data = np.load(infa)
    dataaug = np.load(infb)
    datapitch = np.load(infc)

    fig = plt.figure()

    idx = random.randint(0, len(data)-1)
    fig.add_subplot(6, 1, 1)
    plt.imshow(data[idx].T)
    fig.add_subplot(6, 1, 2)
    plt.imshow(dataaug[idx].T)
    fig.add_subplot(6, 1, 3)
    plt.imshow(datapitch[idx].T)

    idx = random.randint(0, len(data)-1)
    fig.add_subplot(6, 1, 4)
    plt.imshow(data[idx].T)
    fig.add_subplot(6, 1, 5)
    plt.imshow(dataaug[idx].T)
    fig.add_subplot(6, 1, 6)
    plt.imshow(datapitch[idx].T)

    plt.show()


import plac
plac.call(main)
