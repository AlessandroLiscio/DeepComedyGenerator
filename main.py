import time

from utils.preprocessing import *
from utils.results import *
from generator import Generator
# from tokenizer import Tokenizer

#########
# SETUP #
#########

# ATTENTION: assert d_model % self.num_heads == 0

# model hyperparameters
num_layers_encoder = 5
num_layers_decoder = 5
num_heads = 4
d_model = 256
dff = 512
dropout_rate = 0.2

# one epoch is all you need
epochs_production = 0
epochs_comedy = 1

# number of repetitions per dataset, instead of epochs
repetitions_production = 0
repetitions_comedy = 10 #70

# append files' names in the desired order
myorder = []
if epochs_production > 0:
  # Sort files lists in custom order
  production_list = ['tokenized_convivio.txt','tokenized_vita.txt', 'tokenized_detto.txt','tokenized_fiore.txt']
  for filename in production_list:
    myorder.append(filename)
if epochs_comedy > 0:
  myorder.append('tokenized_commedia.txt')

in_path = "data/"
out_path = "results/"

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
vocab_size, str2idx, idx2str = create_vocab(files, myorder)

# Print files' names and texts
print("\n{}\n".format('='*45))
print("myorder: ", myorder)
print("Files: ", len(files))
print("Files names:")
for i, file_name in enumerate(files):
  print("\t{}- {}".format(i+1, file_name))
print("\n{}\n".format('='*45))

#####################
# DATASETS CREATION #
#####################

# Production dataset
if epochs_production > 0:

    # Create Dante's Production datasets list
    dataset_production = []
    for file_name in myorder:
      if not file_name == "tokenized_commedia.txt":
        dataset_production.append(files[file_name])

    # Split input target for Dante's Production dataset
    print("Generating Dante's Production")
    dataset_production, real_size_production = split_input_target_production(
      dataset_production, str2idx, inp_len = 3, tar_len = 3, repetitions = repetitions_production)
    print("Real size production: ", real_size_production)

# Comedy dataset
if epochs_comedy > 0:

    # Split input target for Divine Comedy dataset
    print("Generating Divine Comedy")
    dataset_comedy, max_len, real_size_comedy = split_input_target_comedy(
          files["tokenized_commedia.txt"], str2idx, inp_len = 3, tar_len = 4, repetitions = repetitions_comedy)
    print("Real size comedy: ", real_size_comedy)

# Print samples of the generated Comedy dataset
for (batch, (inputs, targets)) in enumerate(dataset_comedy.take(1)):
  print("\n{} [ Dataset Sample ] {}\n".format("="*13, "="*13))
  print("-- input:\n\n{}\n\n-- target:\n\n{}\n".format(clear_text(ints_to_text(inputs[0], idx2str)),clear_text(ints_to_text(targets[0], idx2str))))
  print("{}".format("="*45))

#############
# TOKENIZER #
#############

# from tokenizer import Tokenizer
# tokenizer = Tokenizer()
# tokenizer.tokenize("path/to/divine_comedy.txt")

#############
# GENERATOR #
#############

generator = Generator(vocab_size = vocab_size,
                      encoders = 5, 
                      decoders = 5, 
                      d_model = 256,
                      dff = 512,
                      heads = 4,
                      dropout = 0.2)

print(generator)

############
# TRAINING #
############

# Train on Dante's production
if epochs_production > 0:
  t_production, loss_hist_production, acc_hist_production = generator.train_model(dataset_production, epochs_production, real_size_production)
else:
  t_production = 0
  loss_hist_production = ["0"]
  acc_hist_production = ["0"]

# Train on divine comedy
if epochs_comedy > 0:
  t_comedy, loss_hist_comedy, acc_hist_comedy = generator.train_model(dataset_comedy, epochs_comedy, real_size_comedy)
else:
  t_comedy = 0
  loss_hist_comedy = ["0"]
  acc_hist_comedy = ["0"]

##############
# GENERATION #
##############

# initialize start string
divine_comedy = files_list[files_names.index("tokenized_commedia.txt")]
print(divine_comedy)
start = list(tf.keras.preprocessing.sequence.pad_sequences([flatten(encode_tokens(split_tokens(divine_comedy[:3]), str2idx))], maxlen=max_len)[0])
print("Start:\n", np.array(divine_comedy)[:3])

# initialize list of generations
generations = []

# choose the list of temperatures (one generation for each temperature)
temperatures = np.round(np.linspace(0.5, 1.5, num=11), 1)

# generate a cantica for each temperature
print("\nGenerating new cantica: ")
for temp in temperatures:

  # start timer
  t_start = time.time()
  print(f"- temperature {temp}... ", end="")

  # generate cantica
  generated_string = generator.generate(str2idx = str2idx,
                                        start = start,
                                        eov = str2idx['</v>'],
                                        max_len = max_len,
                                        max_iterations=100,
                                        temperature=temp)

  # decode the generated cantica and remove special tokens
  generated_string = clear_text(ints_to_text(generated_string, idx2str))

  # stop timer
  t_gen = round(time.time() - t_start)
  print(f"completed ({int(t_gen/3600)}h {int(t_gen/60%60)}m {int(t_gen%60)}s)")

  # append generated cantica to results
  generations.append(generated_string)

#######
# LOG #
#######

# stringify the model description for the file name
model_description = f"{generator.encoders}_{generator.decoders}_{generator.d_model}_{generator.dff}_{generator.heads}_{repetitions_production}_{repetitions_comedy}"

# create the log dictionary
log = {
  "model": {
    "encoders": generator.encoders,
    "decoders": generator.decoders,
    "d_model": generator.d_model,
    "dff": generator.dff,
    "heads": generator.heads
    },
  "trainings": {
    "production": {
        "repetitions": repetitions_production,
        "time": t_production,
        "loss_history": loss_hist_production,
        "acc_history": acc_hist_production
    },
    "comedy": {
        "repetitions": repetitions_comedy,
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

# print model summary
transformer.summary()

# print training information
show_train_info(log)

# print generations in tabular form
show_generations(log, temperatures)

# plot loss and accuracy histories
plot_hist(loss_history, acc_history)
