import json
from utils.preprocessing import *
from generator import Generator
# from tokenizer import Tokenizer

##################
# TRAINING SETUP #
##################

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
repetitions_comedy = 70

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

# Create Dante's Produciton datasets list
dataset_production = []
for file_name in myorder:
  if not file_name == "tokenized_commedia.txt":
    dataset_production.append(files[file_name])
  
# Split input target for Dante's Production dataset
if epochs_production > 0:
  print("Generating Dante's Production")
  dataset_production, real_size_production = split_input_target_production(
      dataset_production, str2idx, inp_len = 3, tar_len = 3, repetitions = repetitions_production)
  print("Real size production: ", real_size_production)

# Split input target for Divine Comedy dataset
if epochs_comedy > 0:
  print("Generating Divine Comedy")
  dataset_comedy, max_len, real_size_comedy = split_input_target_comedy(
          files["tokenized_commedia.txt"], str2idx, inp_len = 3, tar_len = 4, repetitions = repetitions_comedy)
  print("Real size comedy: ", real_size_comedy)

# Print samples of the generated Comedy dataset
print("Comedy datasets Samples:\n")
for (batch, (inputs, targets)) in enumerate(dataset_comedy.take(1)):
  print("{} [batch: {}] {}".format("="*16, batch, "="*16))
  print("-- input:\n\n{}\n\n-- target:\n\n{}\n".format(clear_text(ints_to_text(inputs[0], idx2str)),clear_text(ints_to_text(targets[0], idx2str))))
  print("{}".format("="*45)

#############
# TOKENIZER #
#############

# from tokenizer import Tokenizer
# tokenizer = Tokenizer()
# tokenizer.tokenize("path/to/divine_comedy.txt")

#############
# GENERATOR #
#############

from generator import Generator
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
start = list(tf.keras.preprocessing.sequence.pad_sequences([flatten(encode_tokens(split_tokens(divine_comedy[:3])))], maxlen=max_len)[0])
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
  generated_string = generator.generate(start,
                                        eov = str2idx['</v>'],
                                        max_len = max_len,
                                        temperature=temp,
                                        max_iterations=100)
                                        
  # decode the generated cantica and remove special tokens
  generated_string = clear_text(ints_to_text(generated_string))

  # stop timer
  t_gen = round(time.time() - t_start)
  print(f"completed ({int(t_gen/3600)}h {int(t_gen/60%60)}m {int(t_gen%60)}s)")

  # append generated cantica to results
  generations.append(generated_string)

# stringify the model description for the file name
model_description = f"{generator.encoders}_{generator.decoders}_{generator.d_model}_{generator.dff}_{generator.heads}_{repetitions_production}_{repetitions_comedy}"

# create the log dictionary
log = {
    "model": { 
        "num_layers_encoder": generator.encoder,
        "num_layers_decoder": generator.decoder,
        "d_model": generator.d_model,
        "dff": generator.dff,
        "num_heads": generator.heads
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
for i, temp in enumerate(temperatures):
  log["generations"]["temp_"+str(temp)] = generations[i]

# create destination folder if it doesn't exist
if not os.path.exists(out_path):
  os.mkdir(out_path)
  print("CREATED: ", out_path)

# Save the log file 
log_file = f"{out_path}/LOG_{model_description}.json"
with open(log_file, 'w+') as fp:
  json.dump(log, fp, indent=4)
  print(f"log saved as {log_file}")

# Save the generations as text files
generations_files = []
for temperature, generated_text in zip(log["generations"], generations):
  out_file_name = f"GEN-{temperature}_[{model_description}].txt"
  file_path = f"{out_path}/{out_file_name}"
  with open(file_path, "w+") as out_file:
    out_file.write("\n".join(generated_text[1:]))
    generations_files.append(file_path)
    print(f"generated text at temperature {temperature} saved as {out_file_name}")
print(f"\tin folder {out_path}")

###########
# RESULTS #
###########

# print model summary
transformer.summary()

# print model and training information
print('MODEL:')
for param in log['model']:
  print(f" -- {param}: {log['model'][param]}")
print('\nTRAINING:')
for training in log['trainings']:
  print(f" -- {training}")
  for info in log['trainings'][training]:
    if 'history' in info:
      print(f"   -- {info}: {log['trainings'][training][info][:3]} ... {log['trainings'][training][info][-3:]}")
    elif info == 'time':
      print(f"   -- {info}: {int(log['trainings'][training][info]/3600)}h {int(log['trainings'][training][info]/60%60)}m {int(log['trainings'][training][info]%60)}s")
    else:
      print(f"   -- {info}: {log['trainings'][training][info]}")

#####################
# PRINT GENERATIONS #
#####################

# extract the texts from the log
generations = []
for temp in log['generations']:
  canto = log['generations'][temp] 
  generations.append(canto.replace(' ,',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' :',':').replace(' ;',';').split('\n'))

# header of the table
head_line = "\t    "
for temp in temperatures:
  head_line += "{:<45}".format(temp)
print(head_line+"\n\n")

# organize by columns
for row_idx in range(len(generations[0])):
  row = ""
  for temp_idx in range(len(temperatures)):
    row += "{:<45}".format(generations[temp_idx][row_idx])
  print(row)

####################################
# PLOT LOSS AND ACCURACY HISTORIES #
####################################

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# plot loss and accuracy histories
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

# loss history
loss_history = log['trainings']['comedy']['loss_history']
for i, loss in enumerate(loss_history):
  loss_history[i] = float(loss_history[i])

# accuracy history
acc_history = log['trainings']['comedy']['acc_history']
for i, loss in enumerate(acc_history):
  acc_history[i] = float(acc_history[i]) 

# plot loss history
ax0.set_title('Loss History', color='lightblue', fontsize=15, fontweight= 'bold')
ax0.set_xticks(range(0,len(loss_history),5))
ax0.grid()
ax0.plot(loss_history, color='blue')

# plot accuracy history
ax1.set_title('Accuracy History', color='orange', fontsize=15, fontweight= 'bold')
ax1.set_xticks(range(0,len(acc_history),5))
ax1.set_ylim(top=1)
ax1.grid()
ax1.plot(acc_history, color='red')