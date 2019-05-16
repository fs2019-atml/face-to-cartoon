from .base_model import BaseModel
from . import networks
import numpy as np
import torch

class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode face', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        #### GROUP5 Code ####
        parser.set_defaults(dataset_mode='face')
        parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        #### END of code ####        

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, is_conditional=True)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input, cls_label=1):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real_A = input['A']['img'].to(self.device)
        self.image_paths = input['A_paths']

        ### Get the class label of the cartoon image, and transform it into one-hot
        self.real_B_cls_label = cls_label

        if self.real_B_cls_label < 1 :
            self.real_B_cls_label = 1
        if self.real_B_cls_label > 10:
            self.real_B_cls_label = 10

        cls_num = 10
        # Convert to one-hot format
        self.cls_input = np.zeros((1,10,1,1))
        self.cls_input[0,self.real_B_cls_label - 1,0,0] = 1
        self.cls_input = torch.from_numpy(self.cls_input).float()

        self.cls_input = self.cls_input.to(self.device)


    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A, cls_input=self.cls_input)  # G(A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
