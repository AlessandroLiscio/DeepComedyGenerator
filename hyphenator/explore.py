import json
import os
from hyphenators import *

hyphenator = cnnhyphenator(from_pretrained="hyphenator_model_cnn")

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

# CHECK OUTLIERS

# Initialize Verse Evaluator
VE = VerseEvaluator("our_dict.json")

# Define paths
gen_dir = '../results/generations/'
log_dir = '../results/logs/'

# Create logs directory if it does not exist
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    print("CREATED: ", log_dir)

# If already stressed, don't add "[" and "]"
is_stressed = False

# Define hendecasyllables lower and upper bounds
lb = 11
ub = 11

for filename in os.listdir(gen_dir):
# for filename in ['commedia.txt']:
#     gen_list = open("../data/original/"+filename, 'r').read().replace("|", "").split("\n")

    # Load text file
    gen_list = open(gen_dir+filename, 'r').read().replace("\n\n","\n").split("\n")
    tot = len(gen_list)-1

    # Initialize variables
    hendec_count = 0
    stress_count = 0
    log = ""
    
    print("\n"+filename)

    for i, verse in enumerate(gen_list[:-1]):

        # Remove squares if present
        if is_stressed:
            verse = verse.replace('[', '').replace(']', '')

        # Hypenate verse
        hyphenated_verse = hyphenator.hyphenate(verse)

        # Get stressed syllable position and verse "stress" mask
        try:
            stress_pos, mask = VE.get_stress(hyphenated_verse, return_mask=True)

            # Surround stressed syllable with "[ ]"
            tokens = hyphenated_verse.split("|")
            # Remove empty element and adjust stress_pos
            tokens = tokens[1:]
            stress_pos -= 1
            # print(tokens, len(tokens))
            if len(tokens) in range(lb, ub+1):
                hendec_count += 1
            verse = "|".join(tokens[:stress_pos] + ["["+tokens[stress_pos]+"]"] + tokens[stress_pos+1:])
            
            # Check verse is hendecasyllable
            try: 
                if mask[9] != 1: raise Exception() # catch both cases in which verses have less than 10 syll, and those in which the 10th syll is not stressed
            except: 
                stress_count += 1
            log += "{:<5}  {:60} {:2}\t{}\n".format(i, verse, stress_pos, mask)
            print(f"\r{i+1} / {tot}\t", end="")
        except:
            pass

    print("".join(("\n", "non stressed 10th syllables ", str(stress_count), "/", str(tot))))
    print("".join((f"hendecasyllables ({lb}-{ub}) ", str(hendec_count), "/", str(tot), "\n", "_"*40)))

    # Save log
    with open(log_dir+filename.replace(".txt", ".log"), 'w+') as log_file:
        log_file.write(log)