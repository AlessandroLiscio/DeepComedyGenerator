from utils.preprocessing import *
from utils.results import *
from utils.generator import Generator
# from tokenizer import Tokenizer

###################
# PARSE ARGUMENTS #
###################

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--encoders", type=int,
                    help="number of encoders in the generator model")
parser.add_argument("--decoders", type=int,
                    help="number of decoders in the generator model")
parser.add_argument("--heads", type=int,
                    help="number of attention heads in the generator model")
parser.add_argument("--d_model", type=int,
                    help="embedding size in the generator model")
parser.add_argument("--dff", type=int,
                    help="number of neurons in the ff-layers in the generator model")
parser.add_argument("--dropout", type=int,
                    help="dropout rate of the generator model")


parser.add_argument("--epochs_production", type=int,
                    help="number of training epochs on production dataset")
parser.add_argument("--epochs_comedy", type=int,
                    help="number of training epochs on comedy dataset")


parser.add_argument("--in_path", type=str,
                    help="path of the folder containing the input files")
parser.add_argument("--out_path", type=str,
                    help="path of the folder containing the output files")

#TODO: implement verbosity
#TODO: fix prints
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()

#########
# SETUP #
#########

# files paths
if not args.in_path:  in_path  = "data/"
if not args.out_path: out_path = "results/"
# if not args.in_path:  in_path  = '/content/drive/MyDrive/DC-gen/data/'
# if not args.out_path:  out_path  = '/content/drive/MyDrive/DC-gen/results/'

# model hyperparameters
# ATTENTION: assert d_model % heads == 0
if not args.encoders: encoders = 5
if not args.decoders: decoders = 5
if not args.heads:    heads    = 4
if not args.d_model:  d_model  = 256
if not args.dff:      dff      = 512
if not args.dropout:  dropout  = 0.2

# one epoch is all you need: number of repetitions per dataset, instead of epochs
if not args.epochs_production: epochs_production = 0
if not args.epochs_comedy:     epochs_comedy     = 70

# append files' names in the desired training order
train_order = []
if epochs_production > 0:
  # Sort files lists in custom train_order
  production_list = ['tokenized_convivio.txt','tokenized_vita.txt', 'tokenized_detto.txt','tokenized_fiore.txt']
  for filename in production_list:
    train_order.append(filename)
if epochs_comedy > 0:
  train_order.append('tokenized_commedia.txt')

#############
# LOAD DATA #
#############

# Define files' paths
files_list, files_names = read_files(in_path + "tokenized/")

###################
# TEXT PROCESSING #
###################

# Create files dictionary
files = {files_names[i]:files_list[i] for i in range(len(files_names))}

# Create vocabularies
vocab_size, str2idx, idx2str = create_vocab(files, train_order)

# Print files' names and texts
print("\n{}\n".format('='*45))
print("Files: ", len(files))
print("Files names:")
for i, file_name in enumerate(files):
  print("\t{}- {}".format(i+1, file_name))
print("Files train_order: ", train_order)
print("\n{}\n".format('='*45))

#####################
# DATASETS CREATION #
#####################

## Production dataset
if epochs_production > 0:

  # Create Dante's Production datasets list
  dataset_production = []
  for file_name in train_order:
    if not file_name == "tokenized_commedia.txt":
      dataset_production.append(files[file_name])

  # Split input target for Dante's Production dataset
  print("Generating Dante's Production")
  dataset_production, original_length_production = split_input_target_production(
    dataset_production, str2idx, inp_len = 3, tar_len = 3, repetitions = epochs_production)
  print("Real size production: ", original_length_production)

## Comedy dataset
if epochs_comedy > 0:

  # Split input target for Divine Comedy dataset
  print("Generating Divine Comedy")
  dataset_comedy, max_len, original_length_comedy = split_input_target_comedy(
    files["tokenized_commedia.txt"], str2idx, inp_len = 3, tar_len = 4, repetitions = epochs_comedy)
  print("Real size comedy: ", original_length_comedy)

# Print samples of the generated Comedy dataset
for (batch, (inputs, targets)) in enumerate(dataset_comedy.take(1)):
  print("\n{} [ Dataset Sample ] {}\n".format("="*13, "="*13))
  print("-- input:\n\n{}\n\n-- target:\n\n{}".format(clear_text(ints_to_text(inputs[0], idx2str)),clear_text(ints_to_text(targets[0], idx2str))))
  print("{}".format("="*45))

#############
# TOKENIZER #
#############

#TODO: implement tokenizer

# from tokenizer import Tokenizer
# tokenizer = Tokenizer()
# tokenizer.tokenize("path/to/divine_comedy.txt")

#############
# GENERATOR #
#############

generator = Generator(vocab_size = vocab_size,
                      str2idx = str2idx,
                      idx2str = idx2str,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout)

print(generator)

############
# TRAINING #
############

# Train on Dante's production
if epochs_production > 0:
  t_production, loss_hist_production, acc_hist_production = generator.train_model(dataset_production, original_length_production)
else:
  t_production = 0
  loss_hist_production = ["0"]
  acc_hist_production = ["0"]

# Train on divine comedy
if epochs_comedy > 0:
  t_comedy, loss_hist_comedy, acc_hist_comedy = generator.train_model(dataset_comedy, original_length_comedy)
else:
  t_comedy = 0
  loss_hist_comedy = ["0"]
  acc_hist_comedy = ["0"]

# Save weights
generator.save_weights(epochs_comedy, out_path)

##############
# GENERATION #
##############

# Choose starting tercet
dc_start = files_list[files_names.index("tokenized_commedia.txt")][:3]

# Choose the list of temperatures (one generation for each temperature)
temperatures = np.round(np.linspace(0.5, 1.5, num=11), 1)

# Generate one cantica (100 verses) for each temperature, starting from input tercet
generations = generator.generate_from_tercet(dc_start, temperatures, max_len, 100)

###########
# RESULTS #
###########

# create the log dictionary
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

# load generations to dictionary
for i, temp in enumerate(temperatures):
  log["generations"]["temp_"+str(temp)] = generations[i]

# save results to out_path
save_results(log, out_path)

# print training information
show_train_info(log)

# print generations in tabular form and save to file
tabular_generations(log, out_path)

# plot loss and accuracy histories
plot_hist(log, out_path)
