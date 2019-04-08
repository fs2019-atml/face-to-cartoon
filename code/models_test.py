
import dataloader
import models

def testTrainingImages():

    class TestOpts():
        def __init__(self):
            self.batch_size = 5
            self.results_dir = './results'
            self.datasets_dir = '/home/fsb1/git/pytorch-CycleGAN-and-pix2pix/datasets'
            self.dataset = 'apple2orange'
    
    opts = TestOpts()
    loader = dataloader.forTraining(opts)
    model = models.CycleGANModel(opts)
    
    for i, data in enumerate(loader):
        print('batch:', i)
        model.setInput(data)
        model.optimize()
    
if __name__ == '__main__':
    testTrainingImages()