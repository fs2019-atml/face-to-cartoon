import dataLoader

def testTrainingImages():

    class TestOpts():
        def __init__(self):
            self.batch_size = 5
            self.datasets_dir = '/home/fsb1/git/pytorch-CycleGAN-and-pix2pix/datasets'
            self.dataset = 'apple2orange'
    
    opts = TestOpts()
    loaders = dataLoader.forTraining(opts)

    for loader in loaders:
        for batch in loader:
            print('batch', batch.shape)
            break
    
if __name__ == '__main__':
    testTrainingImages()