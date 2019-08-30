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


def filter_data(feats, targs):
    """ Only returns data where target is 0 or 1 """
    fnew, tnew = [], []
    for f, t in zip(feats, targs):
        if t == 0 or t == 1 or t == 2:
            fnew.append(f.numpy())
            tnew.append(t)
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
    def __init__(self, num_heads=1, prefinal_size=256):
        super(NetX, self).__init__()
        logger.info(f'Creating model with {num_heads} heads and prefinal size {prefinal_size}')
        prefinal_dim = prefinal_size
        self.conv1 = nn.Conv2d(1, 32, 5, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 4, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, prefinal_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(prefinal_dim)

        self.fc_xent_pre = nn.Linear(prefinal_dim, prefinal_dim, bias=False)
        self.fc_iic_pre = nn.Linear(prefinal_dim, prefinal_dim, bias=False)

        self.fc_xent = nn.Linear(prefinal_dim, 3)
        self.fc2 = nn.Linear(prefinal_dim, 3)

    def forward(self, x):
        logger.debug(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        logger.debug(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        logger.debug(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        logger.debug(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        logger.debug(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))

        # x_prefinal = F.relu(self.fc_iic(x))
        # x_xent = F.relu(self.fc_xent(x))
        # logger.debug(x_prefinal.shape)
        # x = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc2]
        x_alt = F.softmax(self.fc_xent(self.fc_xent_pre(x)), dim=1)
        # x_alt = [F.softmax(fc(x_prefinal), dim=1) for fc in self.fc2_alt]
        x = F.softmax(self.fc2(self.fc_iic_pre(x)), dim=1)
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
        if 'fc2' not in name:
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
        data, target = filter_data(data, target)
        iter_num = batch_idx + (epoch-1) * num_iters_per_epoch
        # print(target[:2])
        # new_lr = 0.1
        # if iter_num >= 1:
        #     new_lr = 0.01
        #     if iter_num % (10*num_iters_per_epoch) == 0 and epoch != num_epochs:
        #         new_lr *= 10
        if epoch != 1:
            new_lr = start_lr * np.exp(-iter_num/tau)
            for g in optimizer.param_groups:
                g['lr'] = new_lr

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
        loss += xent_loss(output_alt, target)
        loss += xent_loss(output_perturb_alt, target)
        # loss += torch.sum(torch.stack([IID_loss(o, o_perturb) for o, o_perturb in zip(output_alt, output_perturb_alt)]))
        # loss = IID_loss(output, output_perturb)
        # loss += IID_loss(output_alt, output_perturb_alt)
        # logger.info('Loss %f' % loss.item())
        loss.backward()
        optimizer.step()
        sys.stdout.flush()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch {} - LR {:.5f} -\tLoss: {:.6f}'.format(
                epoch, new_lr, loss.item()))
            # print(model.fc2.weight.grad)
            # print(model.fc2.weight)
            # print(model.fc_xent.weight.grad)
            # print(model.fc_xent.weight)
            # print()
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
            data, target = filter_data(data, target)
            cnt += 1
            data = data.to(device)
            target = target.to(device)
            outputs, outputs_alt = model(data)
            xent_accuracy += (outputs_alt.argmax(-1) == target).sum().item() / target.size(0)
            num_heads = len(out_targs)
            # for i in range(num_heads):
            #     out_targs[i].append(outputs[i].argmax(dim=1).cpu())
            out_targs[0].append(outputs.argmax(dim=1).cpu())
            ref_targs.append(target.cpu())
    xent_accuracy /= cnt
    logger.info('Xent accuracy: %s' % str(xent_accuracy))
    out_targs = [torch.cat(out_targs[i]).cpu() for i in range(num_heads)]
    ref_targs = torch.cat(ref_targs)

    accs = []
    num_classes = 3
    classes_done = set()
    for out_targ in out_targs:
        overlaps = np.zeros((num_classes, num_classes), dtype=np.int32)  # ref idx is row, out idx is col
        for i in range(num_classes):  # num classes
            indcs = (ref_targs == i)
            for j in range(num_classes):
                overlap = (((out_targ == j) + indcs) == 2).sum()
                overlaps[i, j] = overlap
        print(overlaps)
        i = np.argmax(overlaps) // num_classes
        j = np.argmax(overlaps) % num_classes
        numdone = 0
        idxmap = np.arange(num_classes)
        while numdone < num_classes:
            logger.info(f'{i} -> {j}')
            assert i not in classes_done
            classes_done.add(i)
            idxmap[j] = i
            overlaps[:, j] = 0
            i = np.argmax(overlaps) // num_classes
            j = np.argmax(overlaps) % num_classes
            numdone += 1

        out_targs_mapped = idxmap[out_targ.numpy()]
        # print(out_targs_mapped[:10])
        acc = (np.sum(ref_targs.numpy() == out_targs_mapped) / ref_targs.size(0))
        accs.append(acc)
    logger.info('Accuracy : %s' % ' '.join([str(acc) for acc in accs]))
    return -np.max(accs)


def initfn(worker_id):
    np.random.seed(worker_id+10)


def train(args, device, lr, prefinal_size):
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

    model = NetX(prefinal_size=prefinal_size)
    # model.fc2.weights = torch.ones(model.fc2.in_features, model.fc2.out_features) * 0.01
    model.to(device)
    preconditioner = None

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9,), weight_decay=0.0)

    acc = None
    model.fc2.bias.data.fill_(0)
    model.fc_xent.bias.data.fill_(0)
    # model.fc2.weight.data.fill_(0.01)
    # model.fc2.weight.data = model.fc_xent.weight.data.clone()
    # nn.init.orthogonal_(model.fc2.weight.data)
    # nn.init.orthogonal_(model.fc_xent.weight.data)
    print(model.fc_xent.weight.shape)
    print(F.cosine_similarity(model.fc_xent.weight, model.fc2.weight, dim=1))
    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, optimizer, preconditioner, epoch, args.epochs)
        acc = test(model, device, test_loader)
    print(F.cosine_similarity(model.fc_xent.weight, model.fc2.weight, dim=1))

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
    return acc


def main():
    # Training settings
    torch.set_printoptions(profile="full")
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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

    # for sz in [5, 6, 7, 32, 33, 63, 64, 93, 110, 112, 256, 512]:
    train(args, device, args.lr, prefinal_size=5)
    # best = fmin(fn=lambda x: train(args, device, x),
    #             space=hp.uniform('x', 0.01, 0.002),
    #             algo=tpe.suggest,
    #             max_evals=10)
    # print(best)

if __name__ == '__main__':
    main()

