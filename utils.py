from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cifar import CIFAR10

import matplotlib.pyplot as plt
import numpy as np

import math
import os
import time
import torch


def get_dataset(config):
    name = config.data.name
    train_batch_size = config.data.train_batch_size
    test_batch_size = config.data.test_batch_size
    image_size = config.img_size
    download = config.data.download

    if name == "cifar10":
        train_dataset = CIFAR10(
            os.path.join(".", "cifar10"),
            train=True,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = CIFAR10(
            os.path.join(".", "cifar10"),
            train=False,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.ToTensor()
                ]
            ),
        )
    elif name == 'celeba':
        raise NotImplementedError
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
    )

    return train_dataset, test_dataset, train_loader, test_loader


def save_latest(model, name, tgt_attr, epoch):
     # Save model
    if os.path.isfile('autoencoder_%s_tgt%d_%d_.pt' % (name, tgt_attr, epoch - 1)):
        os.remove('autoencoder_%s_tgt%d_%d_.pt' % (name, tgt_attr, epoch - 1))
    torch.save(model.state_dict(), 'autoencoder_%s_tgt%d_%d_.pt' % (name, tgt_attr, epoch))


def load_latest():
    files = os.listdir()
    for f in files:
        if f.startswith('autoencoder_i_') and f.endswith('.pt'):
            epoch_start = int(f.split('_')[-2]) + 1
            return f, epoch_start
    return None, None


def get_taus(batch_size, TOPT_CYCLE, device):
    taus = (math.floor(time.time()) % TOPT_CYCLE)
    taus = taus * torch.ones(size=(batch_size,), dtype=torch.int64).to(device)
    taus_onehot = F.one_hot(taus, num_classes=TOPT_CYCLE)
    return taus, taus_onehot


def visualize_vae(inputs_, output_, name, epoch):
     # VISUALIZATION
    n_images = 5

    fig, axes = plt.subplots(nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=(18, 5))
    orig_images = inputs_.detach().cpu().numpy()[:n_images]
    orig_images = np.moveaxis(orig_images, 1, -1)

    decoded_images = output_.detach().cpu().numpy()[:n_images]
    decoded_images = np.moveaxis(decoded_images, 1, -1)

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            ax[i].axis('off')
            ax[i].imshow(img[i])
    # plt.show()
    plt.savefig("%s_epoch%d.png" % (name, epoch))