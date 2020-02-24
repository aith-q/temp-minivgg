from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

from tqdm import tqdm


def loss(rho, k, n):
    p = dist.Binomial(n, rho)
    return -p.log_prob(k.double())


def train_loop(model, dataloader, eval_fun=None):
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iter = tqdm(enumerate(dataloader), total=len(dataloader), unit=' batches')
    for i, sample in iter:
        loss = forward_pass(model, sample)
        iter.set_description("Current loss = {:.2f}".format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if eval_fun:
            eval_fun()

def forward_pass(model, sample):
    rho = model.forward(sample['image'])
    rho = torch.index_select(rho, -1, torch.tensor([0]))
    rho = torch.flatten(rho)
    kn = sample[1]
    return loss(rho, *kn).sum()

