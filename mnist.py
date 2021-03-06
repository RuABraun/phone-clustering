from __future__ import print_function

import argparse
import subprocess as sp
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms.functional as TF
from loguru import logger
from torchvision import datasets, transforms
from hyperopt import fmin, tpe, hp
import random

plt.style.use('ggplot')


def perturb_imagedata(x):
    y = x.clone()
    batch_size = x.size(0)
    assert len(x.shape) == 4

    # sz = torch.randint(4, 7, (batch_size,))
    trans = tv.transforms.RandomAffine(15, (0.2, 0.2,), (0.2, 0.75,))
    # trans = tv.transforms.Compose([
    #     tv.transforms.RandomCrop(28, padding=4),
    #     tv.transforms.ColorJitter(brightness=(0.2, 0.75,))
    # ])
    for i in range(batch_size):
        # szi = sz[i]
        # xpos = torch.randint(4, 24 - szi, (1,))
        # ypos = torch.randint(4, 24 - szi, (1,))
        y[i, 0] = TF.to_tensor(trans(TF.to_pil_image(y[i, 0])))
        # y[i, 0, xpos.item(): xpos.item() + szi, ypos.item(): ypos.item() + szi] = 1.
    noise = torch.randn(batch_size, 1, x.size(2), x.size(3))
    div = torch.randint(20, 30, (batch_size,), dtype=torch.float32).view(batch_size, 1, 1, 1)
    y += noise / div
    return y


def splitdata(feats, targs):
    fnew, tnew = [], []
    for f, t in zip(feats.numpy(), targs):
        if random.random() > 0.5:
            fnew.append(np.pad(f, [(1, 0,), (0, 0), (0, 0)], mode='constant'))
            tnew.append(t)
        else:
            fnew.append(np.pad(f, [(0, 1,), (0, 0), (0, 0)], mode='constant'))
            tnew.append(t+10)
    fnew = torch.Tensor(fnew)
    tnew = torch.Tensor(tnew).long()
    return fnew, tnew


def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k), '{} {} {} {}'.format(bn, k, x_tf_out.size(0), x_tf_out.size(1))

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    return p_i_j


def IID_loss(x_out, x_tf_out, l=2.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    bs, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    # print(p_i_j)
    loss = (- p_i_j * (torch.log(p_i_j) - l * torch.log(p_j) - l * torch.log(p_i))).sum()
    return loss


class NetX(nn.Module):
    def __init__(self, num_heads=5):
        super(NetX, self).__init__()
        std_dim = 128
        final_fcdim = 256
        self.conv1 = nn.Conv2d(2, std_dim, 5, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(std_dim)
        self.conv2 = nn.Conv2d(std_dim, std_dim, 5, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(std_dim)
        self.conv3 = nn.Conv2d(std_dim, std_dim, 5, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(std_dim)
        self.conv4 = nn.Conv2d(std_dim, std_dim, 4, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(std_dim)
        self.fc1 = nn.Linear(std_dim, final_fcdim, bias=False)
        self.bn_fc = nn.BatchNorm1d(final_fcdim)

        self.fc_xent_pre = nn.Linear(final_fcdim, final_fcdim, bias=False)
        self.fc_iic_pre = nn.Linear(final_fcdim, final_fcdim, bias=False)

        # self.fc2 = nn.ModuleList([nn.Linear(512, 10) for _ in range(5)])
        self.fc_xent = nn.Linear(final_fcdim, 10)
        # self.fc2_alt = nn.ModuleList([nn.Linear(512, 20) for _ in range(5)])
        self.fc2 = nn.Linear(final_fcdim, 20)
        # self.fc2_alt = nn.Linear(512, 20)

    def forward(self, xin):
        logger.debug(xin.shape)
        bs = xin.size(0)
        x = F.relu(self.bn1(self.conv1(xin)))
        logger.debug(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        logger.debug(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        logger.debug(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        logger.debug(x.shape)
        x = x.view(bs, -1)
        x_prefinal = F.relu(self.bn_fc(self.fc1(x)))

        # x_xent = F.relu(self.fc_xent_a(x))
        # logger.debug(x_prefinal.shape)
        # x = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc2]
        x_alt = F.softmax(self.fc_xent(self.fc_xent_pre(x_prefinal)), dim=1)
        # x_alt = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc2_alt]
        x = F.softmax(self.fc2(self.fc_iic_pre(x_prefinal)), dim=1)
        # x_alt = F.softmax(self.fc2_alt(x_prefinal), dim=1)
        return x, x_alt


# def update_lr(iter_num, iters_per_epoch, num_epoch_period=3, min_lr=0.005, max_lr=0.025):
#     fraction = (iter_num % (iters_per_epoch * num_epoch_period)) / (iters_per_epoch * num_epoch_period)
#     return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * fraction))
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


def train_epoch(args, model, device, train_loader, optimizer, preconditioner, epoch, num_epochs):
    model.train()
    num_iters_per_epoch = len(train_loader)
    total_num_iters = num_iters_per_epoch * num_epochs
    tau = - total_num_iters / np.log(0.1)
    new_lr = args.lr
    start_lr = args.lr
    xent_loss = torch.nn.NLLLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(train_loader):
        iter_num = batch_idx + (epoch-1) * num_iters_per_epoch
        if epoch != 1:
            new_lr = start_lr * np.exp(-iter_num/tau)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        data, targs_new = splitdata(data, target)
        data = data.repeat(2, 1, 1, 1)
        newdata = perturb_imagedata(data)
        # fig = plt.figure()
        # fig.add_subplot(2, 1, 1)
        # plt.imshow(data[5, 0])
        # fig.add_subplot(2, 1, 2)
        # plt.imshow(newdata[5, 0])
        # plt.show()

        data = data.to(device)
        data_perturb = newdata.to(device)
        optimizer.zero_grad()
        output, output_alt = model(data)
        output_perturb, output_perturb_alt = model(data_perturb)

        # loss = torch.sum(torch.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output, output_perturb)]))
        loss = IID_loss(output, output_perturb)
        target = target.repeat(2).to(device)
        loss += 0.1 * xent_loss(output_alt, target)
        loss += 0.1 * xent_loss(output_perturb_alt, target)
        # loss += torch.sum(torch.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output_alt, output_perturb_alt)]))
        # loss = IID_loss(output, output_perturb)
        # loss += IID_loss(output_alt, output_perturb_alt)
        # logger.info('Loss %f' % loss.item())
        loss.backward()
        # preconditioner.step()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch {} - LR {:.5f} -\tLoss: {:.6f}'.format(
                epoch, new_lr, loss.item()))
            # report_gradnorms(model)
        # if batch_idx == len(train_loader) - 1:
        #     losses = [0 for _ in range(5)]
        #     for i in range(5):
        #         losses[i] = IID_loss(output[i], output_perturb[i])
        #     logger.info('Losses: %s' % ' '.join([str(l.item()) for l in losses]))


def test(model, device, test_loader):
    model.eval()
    out_targs = [[] for _ in range(1)]
    ref_targs = []
    cnt = 0
    xent_accuracy = 0.
    num_heads = -1
    with torch.no_grad():
        for data, target in test_loader:
            cnt += 1
            data, targs_new = splitdata(data, target)
            data = data.to(device)
            target = target.to(device)
            outputs, outputs_alt = model(data)
            xent_accuracy += (outputs_alt.argmax(-1) == target).sum().item() / target.size(0)
            num_heads = len(out_targs)
            # for i in range(num_heads):
            #     out_targs[i].append(outputs[i].argmax(dim=1).cpu())
            out_targs[0].append(outputs.argmax(dim=1).cpu())
            ref_targs.append(targs_new.cpu())
    xent_accuracy /= cnt
    logger.info('Xent accuracy: %s' % str(xent_accuracy))
    out_targs = [torch.cat(out_targs[i]).cpu() for i in range(num_heads)]
    ref_targs = torch.cat(ref_targs)
    num_clusters = 20

    accs = []
    for out_targ in out_targs:
        overlaps = np.zeros((num_clusters, num_clusters), dtype=np.int32)  # ref idx is row, out idx is col
        for i in range(num_clusters):  # num classes
            indcs = (ref_targs == i)
            for j in range(num_clusters):
                overlap = (((out_targ == j) + indcs) == 2).sum()
                overlaps[i, j] = overlap
        # print(overlaps)
        i = np.argmax(overlaps) // num_clusters
        j = np.argmax(overlaps) % num_clusters
        numdone = 0
        idxmap = np.arange(num_clusters)
        while numdone < num_clusters:
            idxmap[j] = i
            overlaps[:, j] = 0
            i = np.argmax(overlaps) // num_clusters
            j = np.argmax(overlaps) % num_clusters
            numdone += 1

        out_targs_mapped = idxmap[out_targ.numpy()]
        # print(out_targs_mapped[:10])
        acc = (np.sum(ref_targs.numpy() == out_targs_mapped) / ref_targs.size(0))
        accs.append(acc)
    logger.info('Accuracy : %s' % ' '.join([str(acc) for acc in accs]))
    return -np.max(accs)


def initfn(worker_id):
    np.random.seed(worker_id+10)


def train(args, device, lr):
    args.lr = lr
    kwargs = {'num_workers': 1, 'pin_memory': False, 'worker_init_fn': initfn}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=4096, shuffle=True, **kwargs)

    model = NetX()
    # model.fc2.weights = torch.ones(model.fc2.in_features, model.fc2.out_features) * 0.01
    model.to(device)

    preconditioner = None
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9,), weight_decay=0.0)
    acc = None
    model.fc2.bias.data.fill_(0)
    model.fc_xent.bias.data.fill_(0)
    # model.fc_iic_a.weight.data = model.fc_xent_a.weight.data.clone()
    # nn.init.orthogonal_(model.fc2.weight.data)
    # nn.init.orthogonal_(model.fc_xent.weight.data)
    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, optimizer, preconditioner, epoch, args.epochs)
        acc = test(model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
    return acc


def main():
    # Training settings
    torch.set_printoptions(profile="full")
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    gpustats = sp.check_output("nvidia-smi --format=csv --query-gpu=memory.used,memory.free | grep -v 'free'", shell=True)
    meminfo = [int(e.split()[0]) for e in gpustats.decode().split("\n")[:-1]]
    if meminfo[0] < meminfo[2]:
        device = "cuda:0"
    else:
        device = "cuda:1"
    logger.info(f'Using device {device}')

    train(args, device, args.lr)
    # best = fmin(fn=lambda x: train(args, device, x),
    #             space=hp.uniform('x', 0.01, 0.002),
    #             algo=tpe.suggest,
    #             max_evals=10)
    # print(best)

if __name__ == '__main__':
    main()

