# Something in this manner (not yet working)

from PIL import Image
from os import listdir
from os.path import isdir, isfile, join
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
    
def listFiles(dir):
    assert isdir(dir), 'assert dir %s' % dir
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return files

class ImageDataset(Dataset):
    def __init__(self, files_A, files_B, transform):
        self.files_A = files_A
        self.len_A = len(files_A)
        self.files_B = files_B
        self.len_B = len(files_B)
        self.transform = transform
        
    def __len__(self):
        return max(self.len_A, self.len_B)
    
    def __getitem__(self, idx):
        #TODO: choose one index at random in order to not process the same pairs all the time
        file_A = self.files_A[idx % self.len_A]
        file_B = self.files_B[idx & self.len_B] 
        image_A = Image.open(file_A).convert('RGB')
        image_B = Image.open(file_B).convert('RGB')
        if self.transform != None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            #image = self.transform(image.reshape(None,None,3))
        return {'A': image_A, 'B': image_B}

# global transforms:
transforms = Compose([Resize(64),
                      ToTensor(),
                      Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# return a loader for train{A,B}, or test{A,B}
def forTraining(opts):
    images_a = listFiles('{}/{}/trainA'.format(opts.datasets_dir, opts.dataset))
    images_b = listFiles('{}/{}/trainB'.format(opts.datasets_dir, opts.dataset))
    return createLoaders(images_a, images_b, opts.batch_size, True)

def forTesting(opts):
    images_a = listFiles('{}/{}/testA'.format(opts.datasets_dir, opts.dataset))
    images_b = listFiles('{}/{}/testB'.format(opts.datasets_dir, opts.dataset))
    return createLoaders(images_a, images_b, opts.batch_size, False)

def createLoaders(images_a, images_b, batch_size, shuffle):
    dataset_a = ImageDataset(images_a, images_b, transforms)
    loader = DataLoader(dataset_a, batch_size=batch_size, shuffle=shuffle)
    return loader
    
    
