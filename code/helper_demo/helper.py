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
from util.util import tensor2im

import torch
import torchvision.transforms as transforms

import sys
sys.path.append('../')

def to_image(tensor):
    nparray = tensor2im(tensor)
    im = Image.fromarray(nparray)
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
