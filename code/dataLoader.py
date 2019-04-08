# Something in this manner (not yet working)

from PIL import Image
from os import listdir
from os.path import isdir, isfile, join
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
    
def listFiles(dir):
    assert isdir(dir), 'assert dir %s' % dir
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return files

class ImageDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file).convert('RGB')
        if self.transform != None:
            image = self.transform(image)
            #image = self.transform(image.reshape(None,None,3))
        return (image)

# global transforms:
transforms = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

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
    dataset_a = ImageDataset(images_a, transforms)
    dataset_b = ImageDataset(images_b, transforms)
    loader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=shuffle)
    loader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=shuffle)
    return [loader_a, loader_b]
    
    
