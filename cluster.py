import random
import subprocess as sp
import sys

import numpy as np
import plac
import torch as to
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from loguru import logger

BATCH_SIZE = 512
DEVICE = ""


class DataHolder(to.utils.data.Dataset):
    def __init__(self, data_f, data_aug_f):
        self.data = np.load(data_f)
        self.data_aug = np.load(data_aug_f)
        self.size = self.data.shape[0]
        assert self.size == self.data_aug.shape[0]
        logger.info(f"Num samples: {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        spec = self.data[idx]
        idx = random.randint(1, spec.shape[0]-3)  # so that 0 and end never queried
        f = np.copy(spec[idx - 1 : idx + 2])
        f_aug = np.copy(spec[idx - 1 : idx + 2])
        f_shift = np.copy(spec[idx: idx + 3])
        mat = np.array([f, f_aug, f_shift], dtype=np.float32)
        return mat


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


class Net(nn.Module):
    def __init__(self, num_heads=5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, (1, 40), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.fc_1 = nn.Linear(256 * 3, 1024, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc_2 = nn.Linear(1024, 256, bias=False)
        self.fc_3 = nn.Linear(256, 1024, bias=False)
        self.bn_fc3 = nn.BatchNorm1d(1024)

        self.fc_fin = nn.ModuleList([nn.Linear(1024, 300) for _ in range(5)])
        self.fc_fin_alt = nn.ModuleList([nn.Linear(1024, 600) for _ in range(5)])

    def forward(self, x):
        logger.debug(x.size())
        bs = x.size(0)

        # xview = x.view(bs, 1, x.size(1), x.size(2))
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        logger.debug(x.size())

        x = x.view(bs, -1)
        x_fc1 = F.relu(self.bn_fc1(self.fc_1(x)))
        logger.debug(x.size())
        x_prefinal = F.relu(self.bn_fc3(self.fc_3(self.fc_2(x_fc1))) + x_fc1)

        logger.debug(x.size())
        x = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc_fin]
        x_alt = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc_fin_alt]
        return x, x_alt


def train(num_epochs, lr):

    model = Net()
    optimizer = to.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9))
    dataholder = DataHolder("data_small.npy", "data_aug_small.npy")
    dataloader = to.utils.data.DataLoader(
        dataholder, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    num_iters_per_epoch = len(dataloader)
    model.to(DEVICE)
    for epoch in range(1, num_epochs + 1):
        model.train()
        for j, batch in enumerate(dataloader):
            input_normal = batch[:, 0].repeat(2, 1, 1).to(DEVICE)
            input_aug = batch[:, 1:].reshape(batch.size(0) * 2, -1, batch.size(-1)).to(DEVICE)

            output, output_alt = model(input_normal)
            output_aug, output_aug_alt = model(input_aug)

            loss = to.sum(to.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output, output_aug)]))
            loss += to.sum(to.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output_alt, output_aug_alt)]))

            loss.backward()
            optimizer.step()

            if j % (num_iters_per_epoch // 5) == 0:
                logger.info(
                    "Train Epoch {} - LR {:.5f} -\tLoss: {:.6f}".format(
                        epoch, lr, loss.item()
                    )
                )


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

    train(5, 0.01)


plac.call(main)
