import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

############################ SETUP ############################

## PATHS
out_path = '../results/'

## RUN INFO
comedy_name  = 'comedy_np_is_es'
tokenization = 'spaces' # ['base', 'spaces']
generation = 'sampling' # ['beam_search', 'sampling', None]  

## MODEL PARAMETERS
encoders = 3
decoders = 3
heads    = 2
d_model  = 256
dff      = 512
dropout  = 0.2

## TRAINING INFO
epochs_production = 0
epochs_comedy     = 70
checkpoint        = 10

## VERBOSE
verbose = True

## ASSERTS
assert d_model % heads == 0
assert generation in ['sampling', 'beam_search', None]

######################### OUTPUT FOLDER ###########################

# Create output folder
if not os.path.exists(out_path):
    os.mkdir(out_path)
    print("CREATED: ", out_path)

######################### LOAD PRETRAINED ###########################

dataloader = DataLoader(from_pickle = out_path,
                        comedy_name = comedy_name,
                        tokenization = tokenization)

dataloader.print_comedy_samples(1, text=True, ints=True)

generator = Generator(dataloader = dataloader,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout,
                      verbose = verbose)

########################### GENERATIONS ###########################

# CHOOSE STARTING TERCET
start = dataloader.get_comedy_start()
print("start:\n", np.array(start))

# CHOOSE LIST OF TEMPERATURES (ONE GENERATION FOR EACH TEMPERATURE)
if generation == 'sampling':
  # temperatures = np.round(np.linspace(0.5, 1.25, num=4), 2)
  temperatures = np.round(np.linspace(0.7, 1.3, num=5), 2)
elif generation == 'beam_search':
  temperatures = np.round(np.linspace(1.0, 1.0, num=1), 1)

# START GENERATION
for ckpt_production in range(0, epochs_production+1, checkpoint):
  for ckpt_comedy in range(40, epochs_comedy+1, checkpoint):
    
    generator.epochs['production'] = min(ckpt_production, epochs_production)
    generator.epochs['comedy'] = min(ckpt_comedy, epochs_comedy)

    if os.path.isdir(generator.get_model_folder(out_path)):
        print(f"\n>> RESULTS FOR CHECKPOINT: {generator.epochs['production']}_{generator.epochs['comedy']}")
        generator.load(out_path, verbose=False)
        log = generator.generate_from_tercet(start, temperatures, 100, generation)
        generator.save_generations(out_path, verbose=False)
        generator.generations_table(out_path, verbose=False)