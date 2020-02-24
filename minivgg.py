from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np

minicfg = [32, 32, 'M', 32, 32, 'M', 16, 16, 'M']
microcfg = [32, 32, 'M', 16, 'M', 16, 'M']
nanocfg = [8, 8, 'M', 4, 'M', 4, 'M']

class MiniVGG(nn.Module):

    def __init__(self, cfg=minicfg, class_count=2, init_weights=True):
        super(MiniVGG, self).__init__()
        self.features = make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(128, class_count),
            nn.Softmax(dim=1)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # return torch.index_select(x, -1, torch.tensor([0]))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    img_size = 128
    in_channels = 1
    for v in cfg:
        if v == 'M':
            img_size //= 2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.Flatten()]
    layers += [nn.Linear(img_size*img_size * in_channels, 128), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
