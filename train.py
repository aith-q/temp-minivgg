from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist


def loss(rho, k, n):
    p = dist.Binomial(n, rho)
    return -p.log_prob(k)
