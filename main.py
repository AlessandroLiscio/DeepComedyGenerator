from src.parser import Parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

############################ SETUP ############################

## LOCAL
in_path  = 'data/tokenized/verses_sov/'
out_path  = "results/"

# ## SLURM
# in_path  = 'data/tokenized/verses_sov/'
# out_path  = '../../../../../public/liscio.alessandro/results/'

# ## COLAB
# in_path = '/content/drive/MyDrive/DC-gen/data/tokenized/verses_sov/' 
# out_path = '/content/drive/MyDrive/DC-gen/results/'

parser = Parser(in_path=in_path,
                out_path=out_path,
                comedy_name='comedy_11_np_is_es',
                tokenization='spaces', # ['base', 'spaces']
                generation='sampling', # ['sampling', 'beam_search', None]
                encoders=3,
                decoders=3,
                heads=2,
                d_model=256,
                dff=512,
                dropout=0.2,
                epochs_production=0,
                epochs_comedy=70,
                checkpoint=10,
                verbose=True)

############################ ARGS ############################

## PATHS
in_path  = parser.in_path
out_path = parser.out_path

## RUN INFO
comedy_name  = parser.comedy_name
tokenization = parser.tokenization
generation   = parser.generation

## MODEL PARAMETERS
encoders = parser.encoders
decoders = parser.decoders
heads    = parser.heads
d_model  = parser.d_model
dff      = parser.dff
dropout  = parser.dropout

## TRAINING INFO
epochs_production = parser.epochs_production
epochs_comedy     = parser.epochs_comedy
checkpoint        = parser.checkpoint

## VERBOSE
verbose = parser.verbose

## ASSERTS
assert d_model % heads == 0
assert generation in ['sampling', 'beam_search', None]

######################### OUTPUT FOLDER ###########################

# Create output folder
if not os.path.exists(out_path):
    os.mkdir(out_path)
    print("CREATED: ", out_path)

########################### DATALOADER ###########################

if os.path.isfile(f"{out_path}{comedy_name}_{tokenization}/dataloader.pkl"):
  dataloader = DataLoader(from_pickle = out_path,
                        comedy_name = comedy_name,
                        tokenization = tokenization,
                        verbose = verbose)
  print(dataloader)
else:
  dataloader = DataLoader(in_path=in_path,
                          comedy_name=comedy_name,
                          tokenization=tokenization,
                          repetitions_production=epochs_production,
                          repetitions_comedy=epochs_comedy,
                          verbose = verbose)
  dataloader.save(out_path)

############################ GENERATOR ############################

generator = Generator(dataloader = dataloader,
                      encoders = encoders, 
                      decoders = decoders, 
                      d_model = d_model,
                      dff = dff,
                      heads = heads,
                      dropout = dropout,
                      verbose = verbose)

# Print comedy samples
dataloader.print_comedy_samples(1, text=True, ints=True)

# Train model on datasets
generator.train_model(checkpoint = checkpoint,
                      out_path = out_path)

# Print training information
# generator.print_training_info()

########################### GENERATIONS ###########################

if generation:

  # CHOOSE STARTING TERCET
  start = dataloader.get_comedy_start()
  print("start:\n", np.array(start))

  # CHOOSE LIST OF TEMPERATURES (ONE GENERATION FOR EACH TEMPERATURE)
  if generation == 'sampling':
    temperatures = np.round(np.linspace(0.7, 1.3, num=5), 2)
  elif generation == 'beam_search':
    temperatures = np.round(np.linspace(1.0, 1.0, num=1), 1)

  # START GENERATION
  for ckpt_production in range(0, epochs_production+1, checkpoint):
    for ckpt_comedy in range(0, epochs_comedy+1, checkpoint):
      
      generator.epochs['production'] = min(ckpt_production, epochs_production)
      generator.epochs['comedy'] = min(ckpt_comedy, epochs_comedy)

      if os.path.isdir(generator.get_model_folder(out_path)):
          print(f"\n>> RESULTS FOR CHECKPOINT: {generator.epochs['production']}_{generator.epochs['comedy']}")
          generator.load(out_path, verbose=False)
          log = generator.generate_from_tercet(start, temperatures, 100, generation)
          generator.save_generations(out_path, verbose=False)
          generator.generations_table(out_path, verbose=False)