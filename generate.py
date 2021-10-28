############################ SETUP ############################

runtimes = ['local', 'slurm', 'colab']
runtime = 'local'
comedy_name  = 'comedy_11_np_is_es'
tokenization = 'spaces'

from src.parser import Parser
if runtime in runtimes: parser = Parser(runtime)
else: raise ValueError(f"Incorrect runtime found. Please choose one in {runtimes}.")

############################ ARGS ############################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

## DATASET INFO
if parser.comedy_name:  comedy_name  = parser.comedy_name
if parser.tokenization: tokenization = parser.tokenization

## PATHS
in_path  = parser.in_path
out_path = parsers.out_path

## MODEL PARAMETERS
encoders = parser.encoders
decoders = parser.decoders
heads    = parser.heads
d_model  = parser.d_model
dff      = parser.dff
dropout  = parser.dropout

assert d_model % heads == 0

## TRAINING INFO
epochs_production = parser.epochs_production
epochs_comedy     = parser.epochs_comedy
checkpoint        = parser.checkpoint

## VERBOSE
verbose = parser.verbose

######################### OUTPUT FOLDERS #########################

dataloaders_path = out_path + 'dataloaders/'
# Create output folders
for folder in [out_path, dataloaders_path]:
  if not os.path.exists(out_path):
      os.mkdir(out_path)
      print("CREATED: ", folder)

######################### LOAD PRETRAINED ###########################

dataloader = DataLoader(from_pickle = dataloaders_path,
                        comedy_name = comedy_name,
                        tokenization = tokenization)

generator = Generator(dataloader = dataloader,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout,
                      verbose = verbose)

########################### GENERATIONS ###########################

if not runtime == 'colab': # let's not waste colab precious gpu time

  # Choose starting tercet
  start = dataloader.get_comedy_start()

  # Choose the list of temperatures (one generation for each temperature)
  temperatures = np.round(np.linspace(0.5, 1.5, num=5), 2)
  # temperatures = np.round(np.linspace(0.5, 1.5, num=11), 1)

  for ckpt_production in range(0, epochs_production+1, checkpoint):
    for ckpt_comedy in range(0, epochs_comedy+1, checkpoint):
      
      generator.trained_epochs_production = min(ckpt_production, epochs_production)
      generator.trained_epochs_comedy = min(ckpt_comedy, epochs_comedy)

      # TODO: fix generator.generations_table
      try:
        generator.load_model_weights(out_path, verbose=False)
        print(f"\n>> RESULTS FOR CHECKPOINT: {generator.trained_epochs_production}_{generator.trained_epochs_comedy}")
        log = generator.generate_from_tercet(start, temperatures, 100)
        generator.save_generations(out_path, verbose=False)
        # generator.generations_table(out_path, verbose)

      except:
        continue
