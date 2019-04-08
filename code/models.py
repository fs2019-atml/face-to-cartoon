# Create some first architectures for the Generator{A2B, B2A} and Discriminator{A, B} Network
import itertools
import torch
import torch.nn as nn
import torchvision

from utils import getDevice, setRequiresGrad

class SimpleGenerator(nn.Module):
    
    ''' Simple generator to start with '''
    # TODO: size is hardcoded
    # TODO: use resnet blocks
    
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        width = 64
        self.layers = nn.Sequential(
                nn.Linear(3*width*width, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                
                nn.Linear(512, 512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                
                nn.Linear(512, 3*width*width)              
                )
    
    def forward(self, input):
        width = 64
        out = input.view(input.size(0), 3*width*width)
        out = self.layers(out)
        return out.view(input.size(0), 3, width, width)
        

class SimpleDiscriminator(nn.Module):
    
    ''' two layer conv net to start with '''
    # TODO: size is hardcoded
    
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.conv = nn.Sequential(
                # input: 3x256x256 // (width)
                nn.Conv2d(3, 64, 5),
                # output: 64x252x252 // (width-4)
                #nn.InstanceNorm2d(64*5*5)
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 5),
                # output: 64x248x248 // (width-8)
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2),
                #output: 64x124x124 // (width-8)/2         
                )
        width = int((64-8)/2)
        self.dense = nn.Sequential(
                nn.Linear(64*width*width, 512),
                nn.ReLU(),
                nn.Linear(512, 192),
                nn.ReLU(),
                nn.Linear(192, 1)
                )
        
    def forward(self, input):
        width = int((64-8)/2)
        out = self.conv(input)
        out = out.view(input.size(0), 64*width*width)
        return self.dense(out)
        
    

class CycleGANModel():
    
    def __init__(self, opts):
        self.opts = opts
        self.device = getDevice(opts)
        self.net_G_A = self.defineGenerator() # for inputs of A
        self.net_G_B = self.defineGenerator() # for inputs of B
        self.net_D_A = self.defineDiscriminator() # for inputs of A
        self.net_D_B = self.defineDiscriminator() # for inputs of B
        self.loss_cycle_A = nn.L1Loss() # why not for the cycle
        self.loss_cycle_B = nn.L1Loss()
        self.loss_D_A = nn.BCEWithLogitsLoss() # why not?
        self.loss_D_B = nn.BCEWithLogitsLoss()
        # define optimizers over all generators and discriminators
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_G_A.parameters(), self.net_G_B.parameters()))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()))
    
    def defineGenerator(self):
        return SimpleGenerator()
        
    def defineDiscriminator(self):
        return SimpleDiscriminator()
    
    def dumpImages(self):
        torchvision.utils.save_image(self.rec_A[0,:,:,:], '{}/rec_a.png'.format(self.opts.results_dir))
        torchvision.utils.save_image(self.rec_B[0,:,:,:], '{}/rec_b.png'.format(self.opts.results_dir))
        torchvision.utils.save_image(self.fake_A[0,:,:,:], '{}/fake_a.png'.format(self.opts.results_dir))
        torchvision.utils.save_image(self.fake_B[0,:,:,:], '{}/fake_b.png'.format(self.opts.results_dir))
    
    def setInput(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)
        self.input_length = self.real_A.size(0)
    
    def optimize(self):
        self.net_G_A.train()
        self.net_G_B.train()
        self.net_D_A.train()
        self.net_D_B.train()
        self.fake_B = self.net_G_A(self.real_A)
        self.rec_A = self.net_G_B(self.fake_B)
        self.fake_A = self.net_G_B(self.real_B)
        self.rec_B = self.net_G_A(self.fake_A)
        self.dumpImages()
        
        # train generators:
        setRequiresGrad([self.net_D_A, self.net_D_B], False) # disable discriminators
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.printLossG()
        
        # train discriminators
        setRequiresGrad([self.net_D_A, self.net_D_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.printLossD()
    
    def backward_G(self):
        real_label = torch.tensor(1.0).expand(self.input_length,1)

        # compute loss on generators
        pred_A = self.net_D_A(self.fake_B) # Generator A is good if the fake is 1.0 (real)
        self.loss_A = self.loss_D_A(pred_A, real_label)
        
        pred_B = self.net_D_B(self.fake_A) # Generator B is good if the fake is 1.0 (real)        
        self.loss_B = self.loss_D_B(pred_B, real_label)
        
        # compute loss on reconstruction (cycle)
        self.loss_cyc_A = self.loss_cycle_A(self.rec_A, self.real_A) # reconstruction should match
        self.loss_cyc_B = self.loss_cycle_B(self.rec_B, self.real_B)
        
        lambda_cycle = 0.2
        loss_G = self.loss_A + self.loss_B + self.loss_cyc_A*lambda_cycle + self.loss_cyc_B*lambda_cycle
        loss_G.backward(retain_graph=True)
        
        
    def printLossG(self):
        formatter = '{:0.3f}'
        ga = formatter.format(self.loss_A.item())
        gb = formatter.format(self.loss_B.item())
        ca = formatter.format(self.loss_cyc_A.item())
        cb = formatter.format(self.loss_cyc_B.item())
        print('loss G_A:', ga, 'loss G_B:', gb, 'cycle A:', ca, 'cycle B:', cb)
    
    def backward_D(self):
        real_label = torch.tensor(1.0).expand(self.input_length,1)
        fake_label = torch.tensor(0.0).expand(self.input_length,1)

        # TODO use a pool of fakes (instead of the last one!)
        # Maybe use a more GAN Style to train the discriminator??
        pred_real_D_A = self.net_D_A(self.real_B)
        pred_fake_D_A = self.net_D_A(self.fake_B)
        loss_real_A = self.loss_D_A(pred_real_D_A, real_label)
        loss_fake_A = self.loss_D_A(pred_fake_D_A, fake_label)
        self.loss_disc_A = (loss_real_A + loss_fake_A) * 0.5
        self.loss_disc_A.backward(retain_graph=True)
                
        pred_real_D_B = self.net_D_B(self.real_A)        
        pred_fake_D_B = self.net_D_B(self.fake_A)
        loss_real_B = self.loss_D_B(pred_real_D_B, real_label)
        loss_fake_B = self.loss_D_B(pred_fake_D_B, fake_label)
        self.loss_disc_B = (loss_real_B + loss_fake_B) * 0.5
        self.loss_disc_B.backward()
        
        #loss_D = loss_A + loss_B
        
    def printLossD(self):
        formatter = '{:0.3f}'
        da = formatter.format(self.loss_disc_A.item())
        db = formatter.format(self.loss_disc_B.item())
        print('loss disc A:', da, 'loss disc B:', db)      

    
    def save(self, file_name):
        print('save networks:')
        print('save {0}/{1}_net_G_A.pth'.format(opts.checkpoints_dir, file_name))
        # dump networks
        # under opts.checkpoints_dir + fileName




def createModel(opts):
    model = CycleGANModel(opts)
    return model
