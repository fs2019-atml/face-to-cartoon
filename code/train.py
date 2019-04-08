# Implement training procedure

import options
import dataLoader
import models

'''
Main train script to train our CycleGAN
'''

if __name__ == '__main__':
    opts = options.parseTrainingOptions()
    dataset = dataLoader.forTraining(opts)
    model = models.createModel(opts)
    
    # iterate over epochs
    for epoch in range(opts.n_epochs):
        
        for i, data in enumerate(dataset):
            model.setInput(data)
            model.optimize()
            
        # TODO: assess model
        model.save('latest')
        model.save('epoch_{}'.format(epoch))
        
    
    print(opts.results_dir)
    
    
    