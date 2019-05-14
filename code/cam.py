## GROUP 

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


#### GROUP5 code ####

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
framesize = 600 # squared
grayscale = False

# setup cyclegan model:
opt = TrainOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt.phase = 'test'
opt.epoch = 'latest'

# hack a bit to get it up and running:
opt.isTrain = False
model = create_model(opt)      # create a model given opt.model and other options
model.isTrain = False # fix this
model.setup(opt)               # regular setup: load and print networks; create schedulers

brightness = 0.3

transform = transforms.Compose([torchvision.transforms.functional.hflip,
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize((brightness, brightness, brightness),
                                                (0.5, 0.5, 0.5))])

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
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im

def concatenate(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    result = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        result.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return result

while(True):
    # Capture drop buffer
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()

    # Our operations on the frame come here
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hight, width, depth = frame.shape
    crop_pixel = int((width - hight)/2) # crop square
    cropped_frame = frame[:, crop_pixel:width-crop_pixel]
    resized_frame = cv2.resize(cropped_frame, (framesize, framesize))
    cvframe = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) 
    
    pil_img = Image.fromarray(cvframe)
    img = transform(pil_img)
    img = img.view(1, 3, 256, 256)
    img_A = to_image(img[0, :, :, :])
    img_B = model.gen_B(img)
    #torchvision.utils.save_image(img_B[0, :, :, :], 'comic.png')
    img_B = to_image(img_B[0, :, :, :])
    img_AB = concatenate([img_A, img_B])
    img_AB.save('comic.jpg')
    
    print('comic converted')

    #print(frame.shape, " ", cropped_frame.shape, " ", resized_frame.shape)

    # Display the resulting frame
    #cv2.imshow('frame',resized_frame)
    #if cv2.waitKey(10) == 27: 

    #break  # esc to quit

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#### END of code ####
