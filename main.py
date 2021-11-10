############################ IMPORTS ############################

from src.parser import Parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

############################ SETUP ############################

dataset = 'sov_sot' # one of the folders in "data/tokenized/"
stop = ['</v>'] # generation stopping characters

## LOCAL
in_path  = f'data/tokenized/{dataset}/'
out_path  = "results/"

# ## SLURM
# in_path  = f'data/tokenized/{dataset}/'
# out_path  = '../../../../../public/liscio.alessandro/results/'

# # COLAB
# in_path = f'/content/drive/MyDrive/DC-gen/data/tokenized/{dataset}/' 
# out_path = '/content/drive/MyDrive/DC-gen/results/'

parser = Parser(in_path=in_path,
                out_path=out_path,

                comedy_name='comedy_np', # ['comedy_np', 'comedy_11_np']
                tokenization='base', # ['base', 'es', 'is-es']

                inp_len=3,
                tar_len=4,

                encoders=5,
                decoders=5,
                heads=4,
                d_model=256,
                dff=512,
                dropout=0.2,

                epochs_production=0,
                epochs_comedy=150,
                checkpoint=10,
                padding='pre', #TODO: test 'post'

                weight_eov=1.0, #TODO: test
                weight_sot=1.0, #TODO: test

                verbose=True)

assert parser.d_model % parser.heads == 0

######################### OUTPUT FOLDER ###########################

# Create output folder
if not os.path.exists(parser.out_path):
    os.mkdir(parser.out_path)
    print("CREATED: ", parser.out_path)

########################### DATALOADER ###########################

dataloader = DataLoader(in_path=parser.in_path,
                        comedy_name=parser.comedy_name,
                        tokenization=parser.tokenization,
                        inp_len = parser.inp_len,
                        tar_len = parser.tar_len,
                        repetitions_production=parser.epochs_production,
                        repetitions_comedy=parser.epochs_comedy,
                        padding=parser.padding,
                        verbose = parser.verbose)
dataloader.save(parser.out_path)

############################ GENERATOR ############################

generator = Generator(dataloader = dataloader,
                      encoders = parser.encoders, 
                      decoders = parser.decoders, 
                      d_model = parser.d_model,
                      dff = parser.dff,
                      heads = parser.heads,
                      dropout = parser.dropout,
                      weight_eov = parser.weight_eov,
                      weight_sot = parser.weight_sot,
                      stop = stop,
                      verbose = parser.verbose)

# Print comedy samples
dataloader.print_comedy_samples(1, text=True, ints=True)

# Train model on datasets
generator.train_model(checkpoint = parser.checkpoint,
                      out_path = parser.out_path)

# # Print training information
# generator.print_training_info()

########################### GENERATIONS ###########################

# CHOOSE STARTING TERCET
start = dataloader.get_comedy_start()
print("\nstart:\n", np.array(start))

# START GENERATION
for ckpt_production in range(parser.epochs_production, -1, -parser.checkpoint):
  for ckpt_comedy in range(parser.epochs_comedy, -1, -parser.checkpoint):
    
    generator.epochs['production'] = ckpt_production
    generator.epochs['comedy'] = ckpt_comedy

    for generation_type in ['sampling', 'beam_search']:

      # CHOOSE LIST OF TEMPERATURES (ONE GENERATION FOR EACH TEMPERATURE)
      if generation_type == 'sampling':
        temperatures = np.round(np.linspace(0.5, 1.0, num=5), 1)
      elif generation_type == 'beam_search':
        temperatures = np.round(np.linspace(1.0, 1.0, num=1), 1)

      if os.path.isdir(generator.get_model_folder(parser.out_path)):
          print(f"\n>> RESULTS FOR CHECKPOINT: {generator.epochs['production']}_{generator.epochs['comedy']}")
          generator.load(parser.out_path, verbose=False)
          log = generator.generate_from_tercet(start, temperatures, generation_type, 100)
          generator.save_generations(parser.out_path, generation_type, verbose=False)
          # generator.generations_table(parser.out_path, verbose=False)

########################### END ###########################