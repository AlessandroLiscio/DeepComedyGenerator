import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.generator import Generator
from src.dataloader import DataLoader

############################ SETUP ############################

## DATASET INFO
comedy_name  = 'comedy_np_es'
tokenization = 'base'

## PATHS
in_path  = '../data/tokenized/'
out_path = '../results/'

## MODEL PARAMETERS
encoders = 5
decoders = 5
heads    = 4
d_model  = 256
dff      = 512
dropout  = 0.2

assert d_model % heads == 0

## TRAINING INFO
epochs_production = 0
epochs_comedy     = 50
checkpoint        = 10

## VERBOSE
verbose = True

# Create output folder
if not os.path.exists(out_path):
    os.mkdir(out_path)
    print("CREATED: ", out_path)

########################### DATALOADER ###########################

dataloader = DataLoader(in_path=in_path,
                        comedy_name=comedy_name,
                        tokenization=tokenization,
                        repetitions_production=epochs_production,
                        repetitions_comedy=epochs_comedy,
                        verbose = verbose)

# print(dataloader.vocab)
# dataloader.print_comedy_samples(3, text=True, ints=True)