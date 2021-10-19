import numpy as np
import pandas as pd
#import wandb
import itertools
import tqdm
from hyphenators import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default="no", help="yes: train a new model, no: load the pretrained model")
args = parser.parse_args()
args.train = args.train == "yes"

divina_text_file = '../data/divina_textonly.txt'
divina_mask_file = '../data/divina_syll_mask.txt'
divina_syll_file = '../data/divina_syll_textonly.txt'

divina_text = open(divina_text_file, "r", encoding="utf-8").read().split("\n")
divina_mask = open(divina_mask_file, "r", encoding="utf-8").read().split("\n")

if args.train:
    # # # TRAINING
    hyphenator = cnnhyphenator(divina_text, divina_mask)
    hyphenator.fit()
    hyphenator.save("hyphenator_model_cnn")
    print("model saved.")
else:
    # # # LOADING PRTRAINED
    hyphenator = cnnhyphenator(from_pretrained="hyphenator_model_cnn")
    print("model loaded")  

print("\n\n\n")
hyphenator.print_model_summary()

# test
print("\n"*10)
samples = ["nel mezzo del cammin di nostra vita", "mi ritrovai per una selva oscura", "che la diritta via era smarrita"]
for sample in samples:
    print("\n", sample, "\n -> ", hyphenator.hyphenate(sample))
