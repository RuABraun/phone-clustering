import torch as to
import numpy as np
import plac
from cluster import Net
from loguru import logger
from collections import defaultdict

INPUT_SZ = 5


def main(model_fpath='model.pt', fout='preds'):
    logger.remove()

    model = Net()
    model.load_state_dict(to.load(model_fpath, map_location=lambda storage, loc: storage))
    model.eval()
    model = model.cpu()

    idx2utt = {}
    with open('utt2indcs') as fh:
        for line in fh:
            uttid, nr, idxa, idxb = line.split()
            idx2utt[int(nr)] = uttid + f' {nr}'

    data = np.load('data/data_small.npy')
    print(data.shape[0])
    data = data
    fhw = open(fout, 'w')
    for i, spec in enumerate(data):
        spec = to.from_numpy(spec)
        uttid = idx2utt[i+1]
        fhw.write(uttid)
        for j in range(spec.shape[0]-INPUT_SZ):
            inp = spec[j:j+INPUT_SZ].view(1, 1, INPUT_SZ, 40)
            out, _ = model(inp)
            final_out = to.stack(out).sum(dim=0) / len(out)
            pred = final_out.argmax(dim=-1).squeeze().item()
            fhw.write(f' {pred}')
        fhw.write('\n')
    fhw.close()


plac.call(main)