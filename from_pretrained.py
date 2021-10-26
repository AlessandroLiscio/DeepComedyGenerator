from utils.preprocessing import *
from utils.results import *
from utils.generator import Generator
from utils.dataloader import DataLoader
# from tokenizer import Tokenizer

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

# #TODO: implement verbosity
# #TODO: fix prints
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
# if not args.epochs_comedy:     epochs_comedy     = 70

############################# PATHS #############################

###### LOCAL #####
in_path  = 'data/'
out_path  = 'results/'
weights_path = 'weights/'
dataloaders_path = 'dataloaders/'

##### SLURM #####
# in_path = 'data/'
# out_path  = '../../../../../public/alessandro.liscio/results/'
# weights_path = '../../../../../public/alessandro.liscio/weights/'
# dataloaders_path = '../../../../../public/alessandro.liscio/dataloaders/'

##### DRIVE #####
# in_path  = '/content/drive/MyDrive/DC-gen/data/'
# out_path  = '/content/drive/MyDrive/DC-gen/results/'
# weights_path = '/content/drive/MyDrive/DC-gen/weights/'
# dataloaders_path = '/content/drive/MyDrive/DC-gen/dataloaders/'

# Crete output folders
for folder in [out_path, weights_path, dataloaders_path]:
  create_folder(folder)

#################################################################

#########
# SETUP #
#########

# model hyperparameters
# ATTENTION: assert d_model % heads == 0
encoders = 5
decoders = 5
heads    = 4
d_model  = 256
dff      = 512
dropout  = 0.2

# one epoch is all you need: number of repetitions per dataset, instead of epochs
epochs_production = 0
epochs_comedy     = 75

dataloader = DataLoader(sep = '|',
                        in_path='data/tokenized/',
                        comedy_name='comedy_11',
                        tokenization='punctuationless_spaces',
                        epochs_production=epochs_production,
                        epochs_comedy=epochs_comedy,
                        verbose = True)

dataloader.save(dataloaders_path)

# dataloader.print_comedy_samples(1)

#############
# GENERATOR #
#############

generator = Generator(dataloader = dataloader,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout)

print(generator)

# Load weights
generator.trained_epochs_comedy = 0
generator.load_model_weights(weights_path)

############
# TRAINING #
############

# If not None, model weights are saved every 'ckpt' epochs
ckpt = 25

# Train on Dante's production
if epochs_production > 0:
  t_production, loss_hist_production, acc_hist_production = generator.train_model(
    dataloader.datasets['production'], "production", weights_path, ckpt)
else:
  t_production = 0
  loss_hist_production = ["0"]
  acc_hist_production = ["0"]

# Train on divine comedy
if epochs_comedy > 0:
  t_comedy, loss_hist_comedy, acc_hist_comedy = generator.train_model(
    dataloader.datasets['comedy'], "comedy", weights_path, ckpt)
else:
  t_comedy = 0
  loss_hist_comedy = ["0"]
  acc_hist_comedy = ["0"]

# Save weights
generator.save_model_weights(weights_path)

##############
# GENERATION #
##############

# Choose starting tercet
dc_start = open(in_path+f'tokenized_{comedy_name}_{tokenization}.txt', 'r').read().lower().splitlines()[:3]

# Choose the list of temperatures (one generation for each temperature)
temperatures = np.round(np.linspace(0.5, 1.5, num=11), 1)

# Generate one cantica (100 verses) for each temperature, starting from input tercet
generations = generator.generate_from_tercet(dc_start, temperatures, 100)

###########
# RESULTS #
###########

# Create the log dictionary
log = {
  "model": {
    "encoders": generator.encoders,
    "decoders": generator.decoders,
    "heads": generator.heads,
    "d_model": generator.d_model,
    "dff": generator.dff
    },
  "trainings": {
    "production": {
      "epochs": epochs_production,
      "time": t_production,
      "loss_history": loss_hist_production,
      "acc_history": acc_hist_production
    },
    "comedy": {
      "epochs": epochs_comedy,
      "time": t_comedy,
      "loss_history": loss_hist_comedy,
      "acc_history": acc_hist_comedy
    }
  },
  "generations": {}
}

# Load generations to dictionary
for i, temp in enumerate(temperatures):
  log["generations"]["temp_"+str(temp)] = generations[i]

# Save results to out_path
save_results(log, out_path)

# Print training information
show_train_info(log)

# Plot loss and accuracy histories
plot_hist(log, out_path)

# Print generations in tabular form and save to file
tabular_generations(log, out_path)
