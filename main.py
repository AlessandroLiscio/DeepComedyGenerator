from src.generator import Generator
from src.dataloader import DataLoader
import os

#TODO: implement parse arguments

###################
# PARSE ARGUMENTS #
###################

# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument("--encoders", type=int,
#                     help="number of encoders in the generator model")
# parser.add_argument("--decoders", type=int,
#                     help="number of decoders in the generator model")
# parser.add_argument("--heads", type=int,
#                     help="number of attention heads in the generator model")
# parser.add_argument("--d_model", type=int,
#                     help="embedding size in the generator model")
# parser.add_argument("--dff", type=int,
#                     help="number of neurons in the ff-layers in the generator model")
# parser.add_argument("--dropout", type=int,
#                     help="dropout rate of the generator model")


# parser.add_argument("--epochs_production", type=int,
#                     help="number of training epochs on production dataset")
# parser.add_argument("--epochs_comedy", type=int,
#                     help="number of training epochs on comedy dataset")


# parser.add_argument("--in_path", type=str,
#                     help="path of the folder containing the input files")
# parser.add_argument("--out_path", type=str,
#                     help="path of the folder containing the output files")

# parser.add_argument("-v", "--verbose", action="store_true",
#                     help="increase output verbosity")
# args = parser.parse_args()

# # files paths
# if not args.in_path:  in_path  = "data/"
# if not args.out_path: out_path = "results/"

# # model hyperparameters
# # ATTENTION: assert d_model % heads == 0
# if not args.encoders: encoders = 5
# if not args.decoders: decoders = 5
# if not args.heads:    heads    = 4
# if not args.d_model:  d_model  = 256
# if not args.dff:      dff      = 512
# if not args.dropout:  dropout  = 0.2

# # one epoch is all you need: number of repetitions per dataset, instead of epochs
# if not args.epochs_production: epochs_production = 0
# if not args.epochs_comedy:     epochs_comedy     = 50

############################# SETUP #############################

comedy_name = 'comedy_np_es_is'
tokenization = 'spaces'       # ['base', 'spaces']
epochs_production = 0
epochs_comedy = 50
checkpoint = 10               # [int, None]
verbose = True

# model hyperparameters
# ATTENTION: assert d_model % heads == 0
encoders = 5
decoders = 5
heads    = 4
d_model  = 256
dff      = 512
dropout  = 0.2

###### LOCAL #####
in_path  = 'data/tokenized/'
out_path  = 'results/'

##### SLURM #####
# in_path = 'data/tokenized/'
# out_path  = '../../../../../public/alessandro.liscio/results/'

##### DRIVE #####
# in_path  = '/content/drive/MyDrive/DC-gen/data/tokenized/'
# out_path  = '/content/drive/MyDrive/DC-gen/results/'

weights_path = out_path + 'weights/'
generations_path = out_path + 'generations/'
dataloaders_path = out_path + 'dataloaders/'

# Create output folders
for folder in [out_path, weights_path, generations_path, dataloaders_path]:
  if not os.path.exists(folder):
      os.mkdir(folder)
      print("CREATED: ", folder)

######################### DATALOADER #############################

dataloader = DataLoader(in_path=in_path,
                        comedy_name=comedy_name,
                        tokenization=tokenization,
                        repetitions_production=epochs_production,
                        repetitions_comedy=epochs_comedy,
                        verbose = verbose)

# dataloader.print_comedy_samples(1)
dataloader.save(dataloaders_path)

######################### GENERATOR #############################

generator = Generator(dataloader = dataloader,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout,
                      verbose = verbose)

# Train model on datasets
generator.train_model(checkpoint = checkpoint,
                      out_path = out_path)

# Choose starting tercet
start = dataloader.get_comedy_start()

# Choose the list of temperatures (one generation for each temperature)
temperatures = np.round(np.linspace(0.5, 1.5, num=11), 1)

# Generate one cantica (100 verses) for each temperature, starting from input tercet
log = generator.generate_from_tercet(start, temperatures, 100)

######################### RESULTS #############################

# Print training information
generator.show_train_info()

# Save results to out_path
generator.save_log(out_path, verbose)
generator.save_generations(out_path, verbose)
generator.plot_hist('comedy', out_path, verbose)
generator.tabular_generations(out_path, verbose)
