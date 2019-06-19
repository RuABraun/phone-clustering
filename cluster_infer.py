import torch as to
import numpy as np
import plac
from cluster import Net
from loguru import logger


def main(model_fpath='model.pt', fout='preds'):
    logger.remove()

    model = Net()
    model.load_state_dict(to.load(model_fpath, map_location=lambda storage, loc: storage))
    model.eval()
    model = model.cpu()

    uttids = []
    with open('tmpwork/phoneali.scp') as fh:
        for line in fh:
            uttids.append(line.split()[0])

    data = np.load('data_small.npy')
    data = data[:50]
    fhw = open(fout, 'w')
    for i, spec in enumerate(data):
        spec = to.from_numpy(spec)
        # fhw.write(uttids[i])
        for j in range(spec.shape[0]-2):
            inp = spec[j:j+3].view(1, 1, 3, 40)
            out, _ = model(inp)
            final_out = to.stack(out).sum(dim=0) / len(out)
            pred = final_out.argmax(dim=-1).squeeze().item()
            fhw.write(f' {pred}')
        fhw.write('\n')
    fhw.close()


plac.call(main)