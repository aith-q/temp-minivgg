from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import time
from pathlib import Path
from tqdm import tqdm
from math import *
import re

threads = 4


from csv_columns import all_columns
default_columns = ['total_classifications', 'gz2_class', 'total_votes', 'png_loc']

feature_regexes = [
    r'^t01.*a0[1-3].*count$',
]

patterns = [re.compile(r) for r in feature_regexes]
features = [[s for s in all_columns if r.match(s) ] for r in patterns]
flat_features = [s for f in features for s in f]


class gz2Dataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        t0 = time.time()
        self.values = pd.read_csv(csv_file, usecols=default_columns+flat_features)
        print ('Loading took {:.2f}s'.format(time.time() - t0))

        self.img_dir = img_dir
        self.transform = transform

        t0 = time.time()
        # <--------22---------->                           <3>
        # /Volumes/alpha/gz2/png/587732/587732591714893851.png
        self.values['png_loc'] = img_dir + self.values['png_loc'].str.slice(22, -3) + 'jpg'

        # probably not worth the confusion
        # self.values.rename(columns={"png_loc": "img_loc"}, inplace=True)

        # hard-coded for now
        self.values['sum_t01'] = self.values[features[0]].sum(axis=1)

        print ('Adjusting took {:.2f}s'.format(time.time() - t0))


    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.values['png_loc'].iloc[idx]
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        x = {}
        ans = 1
        for f in features[0]:
            x[ans] = (self.values[f].iloc[idx], self.values['sum_t01'].iloc[idx])
            ans += 1

        sample = {'image': image, **x}

        return sample

csv_filename = 'data/gz2/gz2_classifications_and_subjects.csv'
raw_img_dir = 'data/gz2/gz2/png'
img_dir = 'data/gz2/gz2/final'


# a bit kludgy
def preprocess(input_dir, output_dir, transform):
    assert(output_dir != input_dir)

    flag_file = Path(output_dir) / 'done'

    if flag_file.exists():
        return

    dataset = gz2Dataset(csv_filename, input_dir, transform)

    output_paths = dataset.values['png_loc'].str.slice_replace(stop=len(input_dir), repl = output_dir)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=threads)
    for i, sample in tqdm(enumerate(dataloader), desc='Preprocessing images', total=len(dataloader), unit=' images'):
        path = output_paths.iloc[i]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        io.imsave(path, np.array(sample['image']))
    flag_file.open('a+b').close()

preprocess(
    raw_img_dir,
    img_dir,
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
    ])
)

augmentation_transform = transforms.Compose([
    transforms.ToPILImage('L'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90.0, fill=(0,)), # bug in library necessitates (0,)
    transforms.ColorJitter(contrast=0.02),
    transforms.RandomResizedCrop((128, 128), scale=(0.75, 0.9), ratio=(1, 1)),
    transforms.ToTensor()
])

from minivgg import *
from train import *

dataset = gz2Dataset(csv_filename, img_dir, augmentation_transform)

model = MiniVGG(nanocfg)

print(model)
print(features)


plt.ion()
plt.show()

fig = plt.figure()

def show_some(n, dataloader, model = None):

    plt.clf()

    k = int(sqrt(n))

    # for i, sample in tqdm(enumerate(dataloader), total=n, unit=''):
    for i, sample in enumerate(dataloader):
        if (i == n): break

        ax = plt.subplot((n+k-1) // k, k, i + 1)
        plt.tight_layout()
        if model is None:
            ax.set_title('Sample {}'.format(i))
        else:
            x = model.forward(sample['image'])
            print(x)
            rho = x.flatten().tolist()[0]
            ax.set_title('Is smooth: {:.3f}'.format(rho))
        ax.axis('off')
        plt.imshow(transforms.ToPILImage('L')(sample['image'][0]))
    plt.draw()
    plt.pause(1)

def eval_fun():
    counter = 0
    loader = DataLoader(dataset, batch_size=1)
    while True:
        if counter == 0:
            show_some(9, loader, model)
            counter = 32
        counter -= 1
        yield ()


loader = DataLoader(dataset, batch_size=32)
train_loop(model, loader, None)


