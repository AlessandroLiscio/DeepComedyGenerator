from argparse import ArgumentParser

class Parser(ArgumentParser):

    def __init__(self,
                in_path, out_path,
                comedy_name, tokenization, generation,
                inp_len, tar_len,
                encoders, decoders, heads,
                d_model, dff, dropout,
                epochs_production, epochs_comedy, checkpoint,
                verbose):


        ## PATHS
        self.in_path  = in_path
        self.out_path = out_path

        ## RUN INFO
        self.comedy_name  = comedy_name
        self.tokenization = tokenization
        self.generation   = generation

        ## DATASET INFO
        self.inp_len = inp_len
        self.tar_len = tar_len

        ## MODEL PARAMETERS
        self.encoders = encoders
        self.decoders = decoders
        self.heads    = heads
        self.d_model  = d_model
        self.dff      = dff
        self.dropout  = dropout
        assert self.d_model % self.heads == 0

        ## TRAINING INFO
        self.epochs_production = epochs_production
        self.epochs_comedy     = epochs_comedy
        self.checkpoint        = checkpoint

        ## VERBOSE
        self.verbose = verbose

        if not "content/drive/" in self.in_path:
            super().__init__()
            self.__init_args__()


    def __init_args__(self):

        # Parse input arguments from command line
        self.__add_args__()
        inputs = super().parse_args()

        ## PATHS
        if inputs.in_path:  self.in_path  = inputs.in_path
        if inputs.out_path: self.out_path = inputs.out_path

        ## RUN INFO
        if inputs.comedy_name:  self.comedy_name  = inputs.comedy_name
        if inputs.tokenization: self.tokenization = inputs.tokenization
        if inputs.generation:   self.generation   = inputs.generation

        ## DATASET INFO
        if inputs.inp_len: self.inp_len = inputs.inp_len
        if inputs.tar_len: self.tar_len = inputs.tar_len
        
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

        ## PATHS
        self.add_argument("--in_path", type=str,
                            help="path of the folder containing the input files")
        self.add_argument("--out_path", type=str,
                            help="path of the folder containing the output files")

        ## DATASET INFO
        self.add_argument("--comedy_name", type=str,
                            help="divine comedy filename, without extension")
        self.add_argument("--tokenization", type=str,
                            help="tokenization method. Must be either 'base' or 'spaces'")
        self.add_argument("--generation", type=str,
                            help="generation method. Must be either 'sampling' or 'beam_search'")

        self.add_argument("--inp_len", type=int,
                            help="number of verses input to the model")
        self.add_argument("--tar_len", type=int,
                            help="number of verses target for the model")

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
