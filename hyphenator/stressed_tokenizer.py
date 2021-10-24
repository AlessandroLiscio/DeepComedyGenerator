'''
This script can be used by giving the original comedy file (textonly)
as input. It create a .log file containing the tokenized text with
squares surrounding the 10th syllable.
'''

import pickle
import json
import tqdm as pbar
import os

from hyphenators import *

hyphenator = cnnhyphenator(from_pretrained="hyphenator_model_cnn")
print("model loaded")  

# sample = "mi ritrovai per una selva oscura"
# sample_syll =  "|mi |ri|tro|vai |per |u|na |sel|va o|scu|ra"    #hyphenator.hyphenate(sample)

class VerseEvaluator:

    def __init__(self, dictionary_path):
        self.dictionary = json.load(open(dictionary_path))


    def get_stress(self, sample_syll, return_mask=False):

        punct = "\"'.,;:!?»«-—“”’" # TODO: let's do it better :) 
        sample_syll = "".join([s for s in sample_syll.lower() if s not in punct]).strip()              
        tokens = [s.strip() for s in sample_syll.split("|")[1:]]                        
        last_word_start = [i for i,c in enumerate(sample_syll) if c == " "][-1] + 1
        last_word_syll = sample_syll[last_word_start:]
        last_word = last_word_syll.replace("|", "")
        
        try: 
            stress_pos = len(tokens) + self.dictionary[last_word]["stress_pos"]    # get the position from the vocabulary
        except: 
            for v in ["a", "e", "i", "o", "u"]:
                try: 
                    stress_pos = len(tokens) + self.dictionary[v + last_word]["stress_pos"] # probably the word's start is trucated, try possible starting vowels 
                    break
                except:
                    stress_pos = len(tokens) - 1 # suppose the stress on the penultimo :) 

        if return_mask:
            return stress_pos, [0 if i+1 != stress_pos else 1 for i in range(len(tokens))]
        else:
            return stress_pos

# Define paths
gen_dir = '../data/'
log_dir = './'

# Create logs directory if it does not exist
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    print("CREATED: ", log_dir)

# Initialize Verse Evaluator
VE = VerseEvaluator("our_dict.json")

for filename in os.listdir(gen_dir):

    if filename == 'divina_textonly.txt':

        # Load text file
        gen_list = open(gen_dir+filename, 'r', encoding="utf-8").read().split("\n")
        tot = len(gen_list)

        # Initialize variables
        t_count = 1
        count = 0
        log = ""

        for i, verse in enumerate(gen_list):

            # Remove quotes
            verse = verse.replace('» ', '')
            verse = verse.replace('« ', '')
            verse = verse.replace('“ ', '')
            verse = verse.replace('” ', '')

            # Hypenate verse
            hyphenated_verse = hyphenator.hyphenate(verse)

            # Get stressed syllable position and verse "stress" mask
            stress_pos, mask = VE.get_stress(hyphenated_verse, return_mask=True)

            # Surround stressed syllable with "[ ]"
            tokens = hyphenated_verse.split("|")
            stressed_verse = "|".join(tokens[:stress_pos] + ["["+tokens[stress_pos]+"]"] + tokens[stress_pos+1:])
            
            # Check verse is hendecasyllable
            try: 
                if mask[9] != 1: raise Exception() # catch both cases in which verses have less than 10 syll, and those in which the 10th syll is not stressed
            except: 
                count += 1
            log += stressed_verse+'\n'

            print(f"\r{i+1} / {tot}", end="")

        # Save log
        with open(log_dir+filename.replace(".txt", ".log"), 'w+') as log_file:
            log_file.write(log)