import plac
import numpy as np
import os

from prep_data import MINLEN


def main(alif, dataf):
    data = []
    with open(alif) as fh:
        for line in fh:
            key, *ids = line.split()
            lst = []
            for id in ids:
                id = int(id)
                if id > 10:
                    lst.append(id)
                elif len(lst) > MINLEN:
                    data.extend(lst)
                    lst = []
            if len(lst) > MINLEN:
                data.extend(lst)

    data = np.array(data)
    np.save(dataf, data)
    np.save(os.path.basename(dataf) + '_small.npy', data[:10000])

plac.call(main)