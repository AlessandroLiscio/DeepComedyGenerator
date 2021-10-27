import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

################
# DATA LOADING #
################

# Read files and save them in list_of_lists
def read_files(directory):

  '''reads all .txt files in 'directory' and returns:
      - files_list : list of list of strings, where each
                     list of string is one row of the text file
      - files_names : list of strings'''

  files_list = []
  files_names = []

  for filename in os.listdir(directory):
    if not filename.endswith(".txt"): continue
    else:

      # Read file
      path_to_file = os.path.join(directory, filename)
      text = open(path_to_file, 'r').read().lower()

      # generate list from text by splitting lines
      text_list = text.splitlines()
      files_list.append(text_list)
      files_names.append(filename)
 
  return files_list, files_names

###################
# TEXT PROCESSING #
###################

# Replaces special tokens with more readable ones and remove weird spaces
def clear_text(text):
  if text[0] == " ":
    text = text[1:]
  text = text.replace('  ', ' ')
  text = text.replace('<s>', '|')
  text = text.replace('<v>', '')
  text = text.replace('<t>','\n')
  text = text.replace('</v>', '\n')
  text = text.replace('~', ' ')
  return text

# Flattens list
def flatten(in_list):
  return [token for verse in in_list for token in verse]

# Returns list of lists from list of strings
def split_tokens(list_of_strings, sep='|'):
  return [verse.split(sep) for verse in list_of_strings]

# Encodes tokens by using the vocabulary
def encode_tokens(data, str2idx):
  return [text_list_to_int(verse, str2idx) for verse in data]

# Returns set of syllales from input list of verses
def verses_to_syllables_set(verses_list, sep='|', verbose=False):
  
  syllables = split_tokens(verses_list, sep)
  syllables = flatten(syllables)
  syllables = sorted(set(syllables), key=len)

  if verbose:
    print(syllables)
    print("syllables set: ", len(syllables))

  return syllables

# Returns set of words from input list of verses
def verses_to_words_set(verses_list, sep='|', verbose=False):
  
  words = []

  for verse in verses_list:

    verse = verse.replace("<v>","")
    verse = verse.replace("</v>","")
    verse = verse.replace("<t>","")
    verse = verse.replace(" ","")

    words.append(verse.split(sep))

  words = flatten(words)
  words = sorted(set(words), key=len)

  if verbose:
    print(words)
    print("words set: ", len(words))

  return words

# Create vocabularies and mappings from input list of lists
def create_vocabs(files, myorder, syls_sep:str='|', words_sep:str=' ', verbose:bool=False):

  #############################
  # final vocabulary order:   #
  # - special tokens          #
  # - punctuation             #
  # - non-ending syllables    #
  # - ending syllables        #
  #############################

  syllables = sorted(
    set().union(*[verses_to_syllables_set(files[file_name], syls_sep) for file_name in myorder]),
    key=len)
  words = sorted(
    set().union(*[verses_to_words_set(files[file_name], words_sep) for file_name in myorder]),
    key=len)

  # initialize groups
  special_tokens = []
  punctuation = []
  non_ending_sylls = []
  ending_sylls = []

  for token in syllables:
    # NON-SYLLABLES
    if '<' in token:
      special_tokens.append(token)
    elif len(token) == 1 and not token.isalpha():
      punctuation.append(token)
    # SYLLABLES
    else:
      if not token[-1] == " ":
        non_ending_sylls.append(token)
      else:
        ending_sylls.append(token)

  # sort groups
  special_tokens = sorted(special_tokens, key=len)
  punctuation = sorted(punctuation, key=ord)
  non_ending_sylls = sorted(non_ending_sylls, key=len)
  ending_sylls = sorted(ending_sylls, key=len)

  # sylls = non_ending_sylls + ending_sylls
  # sylls = sorted(sylls, key=lambda x: int(len(x.replace(" ", ""))) + sum(int(ord(c)) for c in x.replace(" ", "")))
  # print("SYLLS\n\n", sylls)

  # create the tokens vocabulary following the order
  tokens_vocab = []
  for group in [special_tokens, punctuation, non_ending_sylls, ending_sylls]:
    tokens_vocab.extend(group)

  # insert the empty string at poistion 0
  if '' in tokens_vocab:
    tokens_vocab.remove('')
  tokens_vocab.insert(0, '')

  # Creating a mapping from unique characters to indices
  str2idx = {u:i for i, u in enumerate(tokens_vocab)}
  idx2str = np.array(tokens_vocab)

  # Store the index of the first syllable
  # (+ 1 is for the '' token, which will be padding too)
  sylls_start = len(special_tokens) + len(punctuation) + 1
  
  # # remove rare words (threshold t)
  # threshold = 15
  # unknowns = [syl for syl in alphas if flattened.count(syl) < threshold]
  # np.delete(idx2str, np.searchsorted(idx2str, unknowns)-1)
  # str2idx.update({unk:0 for unk in unknowns})

  if verbose:
    print("\n##################\n")
    print("\n## SPECIAL_TOKENS ##\n\n", special_tokens)
    print("\n## PUNCTUATION ##\n\n", punctuation)
    print("\n## NON-ENDING SYLLABLES ##\n\n", non_ending_sylls)
    print("\n## ENDING SYLLABLES ##\n\n", ending_sylls)
    print("\n## VOCAB LEN ##\n\n", len(tokens_vocab))
    print("\n##################\n")

  return words, sylls_start, len(tokens_vocab), str2idx, idx2str

# Converts our text values to numeric.
def text_list_to_int(string_list, str2idx):
  return np.array([str2idx[c] for c in string_list])
 
# Converts our numeric values to text.
def ints_to_text(ints, idx2str):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2str[ints])

#####################
# DATASETS CREATION #
#####################

def create_dataset(inputs, targets, batch_size = 64, repetitions = 100):

  '''creates cached and prefetched datasets from 'inputs' and 'targets' lists'''

  # Create dataset from inputs and targets
  dataset = tf.data.Dataset.from_tensor_slices((
      tf.keras.preprocessing.sequence.pad_sequences(inputs), 
      tf.keras.preprocessing.sequence.pad_sequences(targets)))
  # cache the dataset to memory to get a speedup while reading from it.
  dataset = dataset.cache()
  # create batched dataset and shuffle it
  buffer_size = len(dataset)
  dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True
            ).repeat(repetitions).padded_batch(batch_size, drop_remainder=True)
  # This allows later elements to be prepared while the current is being processed.
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return buffer_size, dataset

def split_input_target_comedy(text_list, str2idx, sep='|', inp_len=3, tar_len=4, skip=1, 
                              batch_size=64, repetitions=100):

  '''splits comedy dataset in input-target couples'''
  
  inputs = []
  targets = []

  # Prepare data for model (list of integers)
  dataset = split_tokens(text_list, sep)
  dataset = encode_tokens(dataset, str2idx)
  
  # Split input-target
  for i in range(0, len(dataset)-tar_len, skip):
    inputs.append(flatten(dataset[i:i+inp_len]))
    targets.append(flatten(dataset[i:i+tar_len]))

  # Max length of Divine Comedy verses
  max_len = max(len(x) for x in inputs)

  # Create repeated, shuffled and prefetched dataset
  real_size, dataset = create_dataset(inputs, targets, batch_size, repetitions)
 
  # Real dataset size (not repeated)
  real_size_batched = int(real_size/batch_size)

  return dataset, max_len, real_size_batched

def split_input_target_production(datasets, str2idx, sep='|', inp_len=3, tar_len=3, skip=1, 
                                  batch_size=64, repetitions=100):
  
  '''splits production dataset in input-target couples'''

  datasets_list = []
  inputs = []
  targets = []

  for text_list in datasets:

    # Prepare data for model (list of integers)
    dataset = split_tokens(text_list, sep)
    dataset = encode_tokens(dataset, str2idx)
    
    # Split input-target
    for i in range(0, len(dataset)-tar_len, skip):
      inputs.append(flatten(dataset[i:i+inp_len]))
      targets.append(flatten(dataset[i:i+tar_len]))

  # Create repeated, shuffled and prefetched dataset
  real_size, dataset = create_dataset(inputs, targets, batch_size, repetitions)
 
  # Real dataset size (not repeated)
  real_size_batched = int(real_size/batch_size)

  return dataset, real_size_batched
