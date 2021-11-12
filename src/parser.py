from argparse import ArgumentParser

class Parser(ArgumentParser):

    def __init__(self,
                runtime, from_pretrained, train, generate,
                dataset, stop,
                comedy_name, tokenization,
                inp_len, tar_len,
                encoders, decoders, heads,
                d_model, dff, dropout,
                weight_eov, weight_sot,
                epochs_production, epochs_comedy, checkpoint, padding,
                verbose):

        ## RUN INFO
        self.runtime  = runtime
        self.from_pretrained = from_pretrained
        self.train    = train
        self.generate = generate

        ## DATASET INFO
        self.dataset = dataset
        self.comedy_name  = comedy_name
        self.tokenization = tokenization

        ## DATASET PROCESSING
        self.stop = stop
        self.padding = padding
        self.inp_len = inp_len
        self.tar_len = tar_len

        ## MODEL PARAMETERS
        self.encoders = encoders
        self.decoders = decoders
        self.heads    = heads
        self.d_model  = d_model
        self.dff      = dff
        self.dropout  = dropout

        ## TRAINING INFO
        self.epochs_production = epochs_production
        self.epochs_comedy     = epochs_comedy
        self.checkpoint        = checkpoint
        self.weight_eov        = weight_eov
        self.weight_sot        = weight_sot

        ## VERBOSE
        self.verbose = verbose

        if not runtime == 'colab':
            super().__init__()
            self.__init_args__()

        ## PATHS
        if self.runtime == 'local':
            self.in_path  = f'data/tokenized/{self.dataset}/'
            self.out_path  = "results/"
        elif self.runtime == 'slurm':
            self.in_path  = f'data/tokenized/{self.dataset}/'
            self.out_path  = '../../../../../public/liscio.alessandro/results/'
        elif self.runtime == 'colab':
            self.in_path = f'/content/drive/MyDrive/DC-gen/data/tokenized/{self.dataset}/' 
            self.out_path = '/content/drive/MyDrive/DC-gen/results/'


    def __init_args__(self):

        # Parse input arguments from command line
        self.__add_args__()
        inputs = super().parse_args()

        ## RUN INFO
        if inputs.from_pretrained: self.from_pretrained = inputs.from_pretrained
        if inputs.train:           self.train    = inputs.train
        if inputs.generate:        self.generate = inputs.generate

        ## DATASET INFO
        if inputs.dataset:      self.dataset = inputs.dataset
        if inputs.comedy_name:  self.comedy_name  = inputs.comedy_name
        if inputs.tokenization: self.tokenization = inputs.tokenization

        ## DATASET PROCESSING
        # if inputs.stop:    self.stop    = inputs.stop
        if inputs.padding: self.padding = inputs.padding
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
        if inputs.weight_eov:        self.weight_eov      = inputs.weight_eov
        if inputs.weight_sot:        self.weight_sot      = inputs.weight_sot

        ## VERBOSE
        if inputs.verbose: self.verbose = inputs.verbose

    def __add_args__(self):

        ## RUN INFO
        self.add_argument("--from_pretrained", action="store_true",
                            help="if specified, both model and dataloader will be loaded from highest checkpoint.")
        self.add_argument("--train", action="store_true",
                            help="if specified training will be done.")
        self.add_argument("--generate", action="store_true",
                            help="if specified generations will be done.")

        ## DATASET INFO
        self.add_argument("--dataset", type=str,
                            help="daatset name. Must correspond to one of the folders in 'data/tokenized/'")
        self.add_argument("--comedy_name", type=str,
                            help="divine comedy filename, without extension")
        self.add_argument("--tokenization", type=str,
                            help="tokenization method. Must be either 'base' or 'spaces'")

        ## DATASET PROCESSING
        self.add_argument("--padding", type=str,
                            help="padding position, choose either 'pre' or 'post'")
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
        self.add_argument("--weight_eov", type=float,
                            help="weight of the end-of-verse token during training loss computation")
        self.add_argument("--weight_sot", type=float,
                            help="weight of the start-of-tercet token during training loss computation")

        ## VERBOSE
        self.add_argument("-v", "--verbose", action="store_true",
                            help="increase output verbosity")
