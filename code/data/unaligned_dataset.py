import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import torch
import torchvision.transforms as transforms
import numpy as np


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        #self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'

        
        # read out landmarks files to store paths and landmarks in A:
        self.A_paths = []
        #with open(os.path.join(opt.dataroot, 'lfw_landmark/trainImageList.txt'), mode='r') as f:
        if (opt.phase == 'train'):
            self.A_root = os.path.join(self.opt.dataroot, '099000_landmarks')
            with open(os.path.join(opt.dataroot, '099000_landmarks/landmarks.txt'), mode='r') as f: #use all from front selection
                for line in f:
                    tokens = line.split(' ')
                    tokens[0] = '.' + tokens[0].replace('\\','/') # linux like
                    if (len(self.A_paths) < opt.max_dataset_size and os.path.isfile(os.path.join(self.A_root, tokens[0]))):
                        self.A_paths.append(tokens)
        
        if (opt.phase == 'test'):
            self.A_root = ''
            for path in sorted(make_dataset(os.path.join(opt.dataroot, 'test'), opt.max_dataset_size)):
                tokens = [path, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.A_paths.append(tokens)
        
        #self.dir_B = os.path.join(opt.dataroot, 'cartoon_orange_hair_crop')  # create a path '/path/to/data/trainB'
        self.dir_B = os.path.join(opt.dataroot, 'cartoon10k_png')
        
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        print('lenghtA:', self.A_size, 'lengthB:', self.B_size)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A_wide = self.get_transform_304(isBeni=True)
        self.transform_B_wide = self.get_transform_304(centerCrop=True)
        self.transform_A = self.get_transform()
        self.transform_B = self.get_transform()

    def get_transform_304(self, centerCrop=False, isBeni=False):
        transform_list = []
        method=Image.BICUBIC
        
        if (isBeni):
            transform_list.append(transforms.Resize(304))
            transform_list.append(transforms.CenterCrop(304))

        if (centerCrop):
            transform_list.append(transforms.CenterCrop(304))
        
        # blow up (not required)
        # transform_list.append(transforms.Resize([288, 288], method))
        
        return transforms.Compose(transform_list)

    def get_transform(self, centerCrop=False):
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        
        return transforms.Compose(transform_list)
    
    def get_rand_upperleft(self):
        """ Returns two random values between 0 and 48 (excluded)
        This can be used to crop a image randomly and move the landmarks accordingly"""
        return torch.FloatTensor(2).uniform_(0,48).floor()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_tokens = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_path = os.path.join(self.A_root, A_tokens[0])
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.transform_A_wide(A_img)
        A_rand = self.get_rand_upperleft()
        A_img = transforms.functional.crop(A_img, int(A_rand[1].item()), int(A_rand[0].item()), 256, 256)
        A_landmarks = torch.tensor([[float(A_tokens[1]), float(A_tokens[2])],
                                    [float(A_tokens[3]), float(A_tokens[4])],
                                    [float(A_tokens[5]), float(A_tokens[6])],
                                    [float(A_tokens[7]), float(A_tokens[8])],
                                    [float(A_tokens[9]), float(A_tokens[10])]])
        
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.transform_B_wide(B_img)
        B_rand = self.get_rand_upperleft()
        B_img = transforms.functional.crop(B_img, int(B_rand[1].item()), int(B_rand[0].item()), 256, 256)


        # full size cartoons:
        B_landmarks = torch.tensor([self.ld(205, 261),
                                    self.ld(295, 261),
                                    self.ld(249, 308),
                                    self.ld(224, 332),
                                    self.ld(274, 332)])
        #resize_factor_b = 256.0/500.0

        # cropped red hair:
        #B_landmarks = torch.tensor([self.ld(86, 140),
        #                            self.ld(173, 141),
        #                            self.ld(130, 190),
        #                            self.ld(109, 213),
        #                            self.ld(154, 212)])

        # process landmarks on faces:
        # random crop: 304 -> 256
        A_landmarks -= A_rand
        
        # process landmarks on orange cartoons:
        # center crop: 500 -> 304:
        B_landmarks -= torch.Tensor([98, 98])
        # random crop: 304 -> 256
        B_landmarks -= B_rand
    
        # apply image transformation
        A = {'img': self.transform_A(A_img), 'ld': A_landmarks}
        B = {'img': self.transform_B(B_img), 'ld': B_landmarks}

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def ld(self, x, y):
        ''' jitter a bit around '''
         # resize landmarks accordingly
        #result = np.random.normal(np.array([x,y]))
        #return [result[0], result[1]]
        return [float(x), float(y)]

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
