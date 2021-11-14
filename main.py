############################ IMPORTS ############################

from src.parser import Parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

############################ SETUP ############################

parser = Parser(
  
  ## RUN INFO
  runtime = 'local',         # ['local', 'colab', 'slurm']
  from_pretrained=False,
  train=False,
  generate=False,

  ## DATASET INFO
  dataset='sov-sot',        # one of the folders in "data/tokenized/"
  comedy_name='comedy_np',  # ['comedy_np', 'comedy_11_np']
  tokenization='base',      # ['base', 'es', 'is-es']

  ## DATASET PROCESSING
  stop=['</v>'],            # [['</v>'], ['</t>'], ['</v>', '</t>']]
  padding='pre',            # ['pre', 'post']
  inp_len=3,
  tar_len=4,

  ## MODEL PARAMETERS
  encoders=5,
  decoders=5,
  heads=4,
  d_model=256,
  dff=512,
  dropout=0.2,

  ## TRAINING INFO
  epochs_production=0,
  epochs_comedy=150,
  checkpoint=10,
  weight_eov=1.0,
  weight_sot=1.0,

  ## VEROBSE
  verbose=True)

assert parser.d_model % parser.heads == 0

########################### DATALOADER ###########################

dataloader = DataLoader(
  from_pickle = parser.from_pretrained,
  dataloader_path = parser.out_path,
  data_path = parser.in_path,
  dataset = parser.dataset,
  comedy_name = parser.comedy_name,
  tokenization = parser.tokenization,
  inp_len = parser.inp_len,
  tar_len = parser.tar_len,
  repetitions_production = parser.epochs_production,
  repetitions_comedy = parser.epochs_comedy,
  padding = parser.padding,
  verbose = parser.verbose
  )

############################ GENERATOR ############################

generator = Generator(
  from_pretrained = parser.from_pretrained,
  generator_path = parser.out_path,
  dataloader = dataloader,
  encoders = parser.encoders, 
  decoders = parser.decoders, 
  d_model = parser.d_model,
  dff = parser.dff,
  heads = parser.heads,
  dropout = parser.dropout,
  weight_eov = parser.weight_eov,
  weight_sot = parser.weight_sot,
  stop = parser.stop,
  verbose = parser.verbose
  )

# # Print comedy samples
# generator.dataloader.print_comedy_samples(1, text=True, ints=True)

# Train model on datasets
if parser.train:
  generator.train_model(checkpoint = parser.checkpoint,
                        out_path = parser.out_path)

# # Print training information
# generator.print_training_info()

########################### GENERATIONS ###########################

if parser.generate:
  
  # GET GENERATION STARTING INPUT
  start, tokenized_start = dataloader.get_comedy_start()
  print("\nstart:\n", np.array(start))
  print("\ntokenized_start:\n", np.array(tokenized_start))

  # START GENERATION
  for ckpt_production in range(parser.epochs_production, -1, -parser.checkpoint):
    for ckpt_comedy in range(parser.epochs_comedy, -1, -parser.checkpoint):
      
      # Update generator epochs to retrieve the right checkpoint folder
      generator.epochs['production'] = ckpt_production
      generator.epochs['comedy'] = ckpt_comedy

      if os.path.isdir(generator.get_model_folder(parser.out_path)):
        
        print(f"\n>> RESULTS FOR CHECKPOINT: {generator.epochs['production']}_{generator.epochs['comedy']}")
        generator.load(parser.out_path, verbose=False)

        for generation_type in ['sampling', 'beam_search']:
          
          # CHOOSE LIST OF TEMPERATURES (ONE GENERATION FOR EACH TEMPERATURE)
          if generation_type == 'sampling':
            temperatures = np.round(np.linspace(0.5, 1.0, num=6), 1)
          elif generation_type == 'beam_search':
            temperatures = np.round(np.linspace(1.0, 1.0, num=1), 1)

          print(f"\n> {generation_type.upper()}")
          log = generator.generate(tokenized_start, temperatures, generation_type, 100)
          generator.save_generations(parser.out_path, generation_type, verbose=False)
          # generator.generations_table(parser.out_path, verbose=False)
      break
    break

########################### END ###########################