import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import numpy as np
import cv2
from PIL import Image

import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('../')


def to_image(tensor, nrow=8, padding=2,
             normalize=False, range=None, scale_each=False, pad_value=0):
    """from torchvision utils: converts a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                                       normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def concatenate(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    result = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        result.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return result


def setup_options():
    opt = TrainOptions().parse()
    # hard-code some parameters for demo
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = False
    opt.no_flip = True
    opt.display_id = -1
    opt.phase = 'test'
    opt.epoch = 'latest'
    opt.isTrain = False
    return opt
