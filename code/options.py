# use argparse to implement user options

import argparse

class TrainingParser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # define all arguments
        self.parser.add_argument('--dataset', type=str, default='cartoon', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--name', type=str, default='unnamed_task', help='name your task')
        self.parser.add_argument('--datasets_dir', type=str, default='./datasets', help='Root directory of the diffent datasets.')
        self.parser.add_argument('--results_dir', type=str, default='./results', help='where to put the results')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='the amount of epochs to run')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        
        self.args = self.parser.parse_args();
        

def parseTrainingOptions():
    parser = TrainingParser()
    return parser.args;
