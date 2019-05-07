import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random

import torch
import torchvision.transforms as transforms


#### GROUP5 code ####

class FaceDataset(BaseDataset):
    """
    This dataset class can load unpaired face datasets.

    It requires one dataset of real faces with the corresponding landmarks.
    The cartoon faces have to be in the format of xx_filename.png where xx encodes
    the class of the conditional training.
    
    under --dataroot we assume the following structure:
        '099000_landmarks' a file with the landmarks of the real faces.
        'cartoon_conditional' a folder with all the cartoon faces.
        
    These choices are not the most general ones but easy to use in our context.

    During test time we expect the real faces under <--dataroot>/test.
    
    Authors: mostly group05
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # We assert some options:
        assert opt.input_nc == 3
        assert opt.output_nc == 3
        assert opt.direction == 'AtoB'
        assert opt.load_size == 256
        assert opt.crop_size == 256
        
        # the wide size indicates the wide dimension before the random crop.
        self.wide_size = 280        
        
        # read out landmarks files to store paths and landmarks in A:
        
        ## about the landmark file: ##
        # - each line represents one image (space separated)
        # - the first token is the relative path
        # - the following tokens are the coordinates of 5 landmarks: <x1 y1>
        # - the 5 landmarks are: left eye, right eye, nose, left mouth edge, right mouth edge
        # (definition left: x1 "more left than" x2 <=> x1 < x2, origin: top left corner)
        #############################
        
        self.A_paths = []
        if (opt.phase == 'train'):
            self.A_root = os.path.join(self.opt.dataroot, '099000_landmarks')
            with open(os.path.join(opt.dataroot, '099000_landmarks/landmarks.txt'), mode='r') as f:
                for line in f:
                    tokens = line.split(' ')
                    tokens[0] = '.' + tokens[0].replace('\\','/') # linux like
                    if (len(self.A_paths) < opt.max_dataset_size and os.path.isfile(os.path.join(self.A_root, tokens[0]))):
                        self.A_paths.append(tokens)
        
        if (opt.phase == 'test'):
            self.A_root = ''
            for path in sorted(make_dataset(os.path.join(opt.dataroot, 'test'), opt.max_dataset_size)):
                tokens = [path, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.A_paths.append(tokens)
        
        self.dir_B = os.path.join(opt.dataroot, 'cartoon_conditional')
        
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        print('lenghtA:', self.A_size, 'lengthB:', self.B_size)
        
        self.transform_A_wide = self.get_transform_wide(resize=True)
        self.transform_B_wide = self.get_transform_wide()
        self.transform_A = self.get_transform()
        self.transform_B = self.get_transform()

    def get_transform_wide(self, centerCrop=False, resize=False):
        """
        Transforms the input image to the "big" size of 280
        """        
        transform_list = []
        
        if (resize):
            transform_list.append(transforms.Resize(self.wide_size, Image.BICUBIC))

        if (centerCrop):
            transform_list.append(transforms.CenterCrop(self.wide_size))
        
        return transforms.Compose(transform_list)

    def get_transform(self):
        """
        Transforms the image to a normalized tensor
        """
        transform_list = [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]        
        return transforms.Compose(transform_list)
    
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
        
        # Load and convert image A
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.transform_A_wide(A_img)
        A_rand = self.get_rand_upperleft()
        A_img = self.random_crop(A_img, A_rand)
        A_landmarks = torch.tensor([[float(A_tokens[1]), float(A_tokens[2])],
                                    [float(A_tokens[3]), float(A_tokens[4])],
                                    [float(A_tokens[5]), float(A_tokens[6])],
                                    [float(A_tokens[7]), float(A_tokens[8])],
                                    [float(A_tokens[9]), float(A_tokens[10])]])
        
        # Load and convert image B
        B_img = Image.open(B_path).convert('RGB')
        B_img = self.transform_B_wide(B_img)
        B_rand = self.get_rand_upperleft()
        B_img = self.random_crop(B_img, B_rand)


        # full size cartoon landmarks:
        B_landmarks = torch.tensor([self.ld(205, 261),
                                    self.ld(295, 261),
                                    self.ld(249, 308),
                                    self.ld(224, 332),
                                    self.ld(274, 332)])

        # process landmarks on faces:
        # scale down 304 -> 280:
        A_landmarks *= 280.0/304.0
        # random crop: 280 -> 256        
        A_landmarks -= A_rand
        
        # process landmarks on cartoons:
        # center crop: 500 -> 280:
        B_landmarks -= torch.Tensor([110, 110])
        # random crop: 280 -> 256
        B_landmarks -= B_rand
    
        
        # TODO: add the conditional encoding here to the B stuff
    
        # apply image transformation
        A = {'img': self.transform_A(A_img), 'ld': A_landmarks}
        B = {'img': self.transform_B(B_img), 'ld': B_landmarks}        

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def ld(self, x, y):
        return [float(x), float(y)]
    
    def get_rand_upperleft(self):
        """ Returns two random values between 0 and 24 (excluded)
        This can be used to crop a image randomly and move the landmarks accordingly"""
        return torch.FloatTensor(2).uniform_(0,self.wide_size-256).floor()
    
    def random_crop(self, img, rand):
        return transforms.functional.crop(img, int(rand[1].item()), int(rand[0].item()), 256, 256) # random crop

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


#### END of code ####
