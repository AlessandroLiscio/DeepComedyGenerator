from argparse import ArgumentParser

class Parser(ArgumentParser):

    def __init__(self, runtime:str):

        if runtime not in ['local', 'slurm', 'colab']:
            raise ValueError(f"Incorrect runtime found. Please choose one in {runtimes}.")

        ## DATASET INFO
        self.comedy_name = None
        self.tokenization = None
        
        ## IN_PATHS
        if runtime == 'local':
            self.in_path  = 'data/tokenized/'
        elif runtime == 'slurm':
            self.in_path  = 'data/tokenized/'
        elif runtime == 'colab':
            self.in_path = '/content/drive/MyDrive/DC-gen/data/tokenized/'    
            
        ## OUT_PATHS
        if runtime == 'local':
            self.out_path  = "results/"
        elif runtime == 'slurm':
            self.out_path  = '../../../../../public/liscio.alessandro/results/'
        elif runtime == 'colab':
            self.out_path = '/content/drive/MyDrive/DC-gen/results/'

        ## MODEL PARAMETERS
        self.encoders = 5
        self.decoders = 5
        self.heads    = 4
        self.d_model  = 256
        self.dff      = 512
        self.dropout  = 0.2

        assert self.d_model % self.heads == 0

        ## TRAINING INFO
        self.epochs_production = 0
        self.epochs_comedy     = 50
        self.checkpoint        = 10

        ## VERBOSE
        self.verbose = True

        if not runtime == 'colab':
            super().__init__()
            self.__init_args__(runtime)


    def __init_args__(self, runtime:str):

        if runtime == 'local' or runtime == 'slurm':

            # Parse input arguments from command line
            self.__add_args__()
            inputs = super().parse_args()

            ## DATASET INFO
            self.comedy_name  = inputs.comedy_name
            self.tokenization = inputs.tokenization

            ## PATHS
            if inputs.in_path:  self.in_path  = inputs.in_path
            if inputs.out_path: self.out_path = inputs.out_path
            
            ## MODEL PARAMETERS
            if inputs.encoders: self.encoders = inputs.encoders
            if inputs.decoders: self.decoders = inputs.decoders
            if inputs.heads:    self.heads    = inputs.heads
            if inputs.d_model:  self.d_model  = inputs.d_model
            if inputs.dff:      self.dff      = inputs.dff
            if inputs.dropout:  self.dropout  = inputs.dropout

            ## TRAINING INFO
            if inputs.epochs_production: self.epochs_production = inputs.epochs_production
            if inputs.epochs_comedy:     self.epochs_comedy     = inputs.epochs_comedy
            if inputs.checkpoint:        self.checkpoint        = inputs.checkpoint

            ## VERBOSE
            if inputs.verbose: self.verbose = inputs.verbose

    def __add_args__(self):

        ## DATASET INFO
        self.add_argument("comedy_name", type=str,
                            help="divine comedy filename, without extension")
        self.add_argument("tokenization", type=str,
                            help="tokenization method. Must be either 'base' or 'spaces'")

        ## PATHS
        self.add_argument("--in_path", type=str,
                            help="path of the folder containing the input files")
        self.add_argument("--out_path", type=str,
                            help="path of the folder containing the output files")

        ## MODEL PARAMETERS
        self.add_argument("--encoders", type=int,
                            help="number of encoders in the generator model")
        self.add_argument("--decoders", type=int,
                            help="number of decoders in the generator model")
        self.add_argument("--heads", type=int,
                            help="number of attention heads in the generator model")
        self.add_argument("--d_model", type=int,
                            help="embedding size in the generator model")
        self.add_argument("--dff", type=int,
                            help="number of neurons in the ff-layers in the generator model")
        self.add_argument("--dropout", type=int,
                            help="dropout rate of the generator model")

        ## TRAINING INFO
        self.add_argument("--epochs_production", type=int,
                            help="number of training epochs on production dataset")
        self.add_argument("--epochs_comedy", type=int,
                            help="number of training epochs on comedy dataset")
        self.add_argument("--checkpoint", type=int,
                            help="training checkpoint for saving model weights")

        ## VERBOSE
        self.add_argument("-v", "--verbose", action="store_true",
                            help="increase output verbosity")
