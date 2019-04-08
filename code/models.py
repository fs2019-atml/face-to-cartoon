# Create some first architectures for the Generator{A2B, B2A} and Discriminator{A, B} Network

import torch
import torch.nn as nn

class CycleGANModel():
    
    def __init__(self, opts):
        self.opts = opts
        self.net_G_A = ''
        self.net_G_B = ''
        self.net_D_A = ''
        self.net_D_B = ''
    
    def setInput(data):
        #...
    
    def optimize():
        # train on data
    
    def save(file_name):
        print('save networks:')
        print('save {0}/{1}_net_G_A.pth'.format(opts.checkpoints_dir, file_name))
        # dump networks
        # under opts.checkpoints_dir + fileName




def createModel(opts):
    model = CycleGANModel(opts)
    return model
