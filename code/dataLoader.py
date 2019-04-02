# Something in this manner (not yet working)

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

class Aset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx,:]
        if self.transform != None:
            image = self.transform(image.reshape(None,None,3))
        return (image)

class Bset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx,:]
        if self.transform != None:
            image = self.transform(image.reshape(None,None,3))
        return (image)

      
transforms = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

trainA_set = DigitDataset(None, transforms)
testA_set = DigitDataset(None, transforms)
trainA_loader = DataLoader(None, batch_size=32, shuffle=True)
testA_loader = DataLoader(None, batch_size=32, shuffle=False)

trainB_set = DigitDataset(None, transforms)
testB_set = DigitDataset(None, transforms)
trainB_loader = DataLoader(None, batch_size=32, shuffle=True)
testB_loader = DataLoader(None, batch_size=32, shuffle=False)
