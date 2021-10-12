#############
# TOKENIZER #
#############

from tokenizer import Tokenizer
tokenizer = Tokenizer()
tokenizer.tokenize("path/to/divine_comedy.txt")

#############
# GENERATOR #
#############

from generator import Generator
generator = Generator(self, 
                      vocab_size = vocab_size,
                      encoders = 5, 
                      decoders = 5, 
                      d_model = 256,
                      dff = 512,
                      heads = 4,
                      dropout = 0.2)

####### CALL
t_comedy, loss_hist_comedy, acc_hist_comedy = train_model(dataset_comedy, epochs_comedy, real_size_comedy)

#########################
# SAVE RESULTS TO DRIVE #
#########################

import json

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
model_description = f"{num_layers_encoder}_{num_layers_decoder}_{d_model}_{dff}_{num_heads}_{repetitions_production}_{repetitions_comedy}"

# create the log dictionary
log = {
    "model": { 
        "num_layers_encoder": num_layers_encoder,
        "num_layers_decoder": num_layers_decoder,
        "d_model": d_model,
        "dff": dff,
        "num_heads": num_heads
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
results_folder_path = f"{path}/{model_description}"
if not os.path.exists(results_folder_path):
  os.mkdir(results_folder_path)
  print("CREATED: ", results_folder_path)

# Save the log file 
log_file = f"{results_folder_path}/LOG_{model_description}.json"
with open(log_file, 'w+') as fp:
  json.dump(log, fp, indent=4)
  print(f"log saved as {log_file}")

# Save the generations as text files
generations_files = []
for temperature, generated_text in zip(log["generations"], generations):
  out_file_name = f"GEN-{temperature}_[{model_description}].txt"
  file_path = f"{results_folder_path}/{out_file_name}"
  with open(file_path, "w+") as out_file:
    out_file.write("\n".join(generated_text[1:]))
    generations_files.append(file_path)
    print(f"generated text at temperature {temperature} saved as {out_file_name}")
print(f"\tin folder {results_folder_path}")

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