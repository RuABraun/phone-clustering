import random
import subprocess as sp
import sys

import numpy as np
import plac
import torch as to
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from hyperopt import fmin, tpe, hp
from loguru import logger

BATCH_SIZE = 2048
NUM_HEADS = 5
INPUT_SIZE = 5
DEVICE = ""


class DataHolder(to.utils.data.Dataset):
    def __init__(self, data_f, data_aug_f, data_pitch_f, total_context=INPUT_SIZE):
        self.data = np.load(data_f)[:-1]
        self.data_aug = np.load(data_aug_f)[:-1]
        self.data_pitch = np.load(data_pitch_f)
        self.size = self.data.shape[0]
        self.total_context = total_context
        assert self.size == self.data_aug.shape[0] == self.data_pitch.shape[0], '{} {} {}'.format(self.size, self.data_aug.shape[0], self.data_pitch.shape[0])
        logger.info(f"Num samples: {self.size}")

    def __len__(self):
        return self.size - self.total_context

    def __getitem__(self, idx):
        lr_context = self.total_context // 2
        idx += lr_context
        f = self.data[idx - lr_context: idx + lr_context + 1]
        f_aug = self.data_aug[idx - lr_context: idx + lr_context + 1]
        f_pitch = self.data_pitch[idx - lr_context: idx + lr_context + 1]
        return np.asarray([f, f_aug, f_pitch])


def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    return p_i_j


def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    # has had softmax applied
    bs, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    # print(p_i_j)
    loss = (-p_i_j * (to.log(p_i_j) - to.log(p_j) - to.log(p_i))).sum()
    return loss


class ResBlock(nn.Module):
    def __init__(self, small_dim, large_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(small_dim, large_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(large_dim)
        self.fc2 = nn.Linear(large_dim, small_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(small_dim)

    def forward(self, x):
        return F.relu(self.bn2(self.fc2(F.relu(self.bn1(self.fc1(x)))))) + 0.66 * x


class Net(nn.Module):
    def __init__(self, num_heads=NUM_HEADS, num_res_blocks=3):
        super(Net, self).__init__()
        self.bnin = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 256, (3, 40), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, (1, 1), 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(256 * 3, 256, bias=False)

        self.fc_bypass = nn.Linear(3 * 256, 256, bias=False)
        self.bn_bypass = nn.BatchNorm1d(256)

        self.blocks = nn.ModuleList([ResBlock(256, 1024) for _ in range(num_res_blocks)])
        self.fc_prefinal = nn.Linear(256, 512, bias=False)
        self.bn_prefinal = nn.BatchNorm1d(512)

        self.fc_fin = nn.ModuleList([nn.Linear(512, 192) for _ in range(num_heads)])
        self.fc_fin_alt = nn.ModuleList([nn.Linear(512, 400) for _ in range(num_heads)])

    def forward(self, xin):
        logger.debug(xin.shape)
        bs = xin.shape[0]
        xin = self.bnin(xin)

        xin = F.relu(self.bn1(self.conv1(xin)))
        logger.debug(xin.shape)

        x = F.relu(self.bn2(self.conv2(xin)))
        x = x.view(bs, -1)
        xin = xin.view(bs, -1)
        x_hidden = self.fc2(x) + self.bn_bypass(self.fc_bypass(xin))

        logger.debug(x_hidden.shape)
        for block in self.blocks:
            x_hidden = block(x_hidden)
        x_hidden = F.relu(self.bn_prefinal(self.fc_prefinal(x_hidden)))

        logger.debug(x.shape)
        x = [F.softmax(fc(x_hidden), dim=1) for fc in self.fc_fin]
        x_alt = [F.softmax(fc(x_hidden), dim=1) for fc in self.fc_fin_alt if not isinstance(fc, list)]
        return x, x_alt


def report_gradnorms(model):
    minnorm = 10
    maxnorm = 0
    Wnamemin = None
    Wnamemax = None
    maxbnnorm = 0
    BNnamemax = None
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        gradnorm = torch.norm(p.grad).item()
        if 'bn' in name:
            if gradnorm > maxbnnorm:
                maxbnnorm = gradnorm
                BNnamemax = name
            continue
        if gradnorm < minnorm:
            minnorm = gradnorm
            Wnamemin = name
        if gradnorm > maxnorm:
            maxnorm = gradnorm
            Wnamemax = name
    logger.info('minnorm: %s %f\tmaxnorm: %s %f\tBN: %s %f' % (Wnamemin, minnorm, Wnamemax, maxnorm, BNnamemax, maxbnnorm))


def train(num_epochs, lr):

    model = Net()
    optimizer = to.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9))
    # optimizer = to.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dataholder = DataHolder("data/data_flat.npy", "data/data_aug_flat.npy", "data/data_pitch_flat.npy")
    dataloader = to.utils.data.DataLoader(
        dataholder, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    num_iters_per_epoch = len(dataloader)
    model.to(DEVICE)
    avg_loss = 0.
    # model_params = model.named_parameters()
    for epoch in range(1, num_epochs + 1):
        model.train()
        for j, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_normal = batch[:, 0].repeat(2, 1, 1).to(DEVICE)
            input_aug = batch[:, 1:].reshape(batch.size(0)*2, INPUT_SIZE, 40).to(DEVICE)
            logger.debug(input_normal.size())
            logger.debug(input_aug.size())
            input_normal = input_normal.unsqueeze(1)
            input_aug = input_aug.unsqueeze(1)

            output, output_alt = model(input_normal)
            output_aug, output_aug_alt = model(input_aug)

            loss = to.sum(to.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output, output_aug)]))
            loss += to.sum(to.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output_alt, output_aug_alt)]))

            loss.backward()
            # report_gradnorms(model)
            optimizer.step()

            loss = loss.item()
            avg_loss = 0.8*avg_loss + 0.2*loss
            if j % (num_iters_per_epoch // 5) == 0:
                logger.info(
                    "Train Epoch {} - LR {:.5f} -\tLoss: {:.6f}".format(
                        epoch, lr, avg_loss
                    )
                )

    to.save(model.state_dict(), 'model.pt')
    return loss


def main(debug: ("Print debug messages", "flag", "d")):
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    to.manual_seed(0)
    np.random.seed(0)
    to.cuda.manual_seed(0)
    random.seed(0)
    to.backends.cudnn.deterministic = True

    gpustats = sp.check_output(
        "nvidia-smi --format=csv --query-gpu=memory.used,memory.free | grep -v 'free'",
        shell=True,
    )
    meminfo = [int(e.split()[0]) for e in gpustats.decode().split("\n")[:-1]]
    global DEVICE
    if meminfo[0] < meminfo[2]:
        DEVICE = "cuda:0"
    else:
        DEVICE = "cuda:1"
    logger.info(f"Using device {DEVICE}")

    train(3, 0.005)
    # best = fmin(fn=lambda x: train(3, x),
    #             space=hp.uniform('x', 0.003, 0.01),
    #             algo=tpe.suggest,
    #             max_evals=5)
    # print(best)


if __name__ == '__main__':
    plac.call(main)
