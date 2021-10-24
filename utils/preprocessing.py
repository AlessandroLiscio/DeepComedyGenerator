import os
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

# Flattens list
def flatten(in_list):
  return [token for verse in in_list for token in verse]

# Replaces special tokens with more readable ones
def clear_text(text):
  return text.replace('<s>', '|').replace('<v>', '').replace('<t>','\n').replace('</v>', '\n').replace('~', ' ')

# Creates list of tokens from input string with tokens
def split_tokens(data, sep='<s>'):
  split = []
  for verse in data:
    split.append(verse.split(sep))
  return split

# Encodes tokens by using the vocabulary
def encode_tokens(data, str2idx):
  encoded = []
  for verse in data:
    encoded.append(text_list_to_int(verse, str2idx))
  return encoded

def verses_to_syllables_set(verses_list, sep='<s>', verbose=False):

  syllables = [verse.split(sep) for verse in verses_list]
  syllables = flatten(syllables)
  syllables = sorted(set(syllables), key=len)

  if verbose:
    print(syllables)
    print("syllables set: ", len(syllables))

  return syllables

def verses_to_words_set(verses_list, sep='<s>', verbose=False):
  
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

# Create vocabulary and mappings from input list of lists
def create_vocabs(files, myorder, syls_sep:str='<s>', words_sep:str='<s>', verbose:bool=False):

  syllables = sorted(
    set().union(*[verses_to_syllables_set(files[file_name], syls_sep, verbose=True) for file_name in myorder]),
    key=len)
  words = sorted(
    set().union(*[verses_to_words_set(files[file_name], words_sep) for file_name in myorder]),
    key=len)

  # split alphas from non-alphas
  non_alphas = []
  alphas = []
  for token in syllables:
    if not token.isalpha():
      non_alphas.append(token)
    else:
      alphas.append(token)

  # split non-alphas in punctuation, 
  # punctuated words and special tokens
  punctuation = []
  punctuated_syls = []
  special_tokens = []
  for token in non_alphas:
    if len(token) == 1:
      punctuation.append(token)
    elif '<' in token:
      special_tokens.append(token)
    else:
      punctuated_syls.append(token)

  # follow with the rest of the tokens
  alphas.extend(punctuated_syls)
  alphas = sorted(alphas, key=len)
  syls_vocab = []
  for group in [special_tokens, punctuation, alphas]:
    syls_vocab.extend(group)

  # if missing, add empty string at position 0
  if '' in syls_vocab:
    syls_vocab.remove('')
  syls_vocab.insert(0, '')

  # Creating a mapping from unique characters to indices
  str2idx = {u:i for i, u in enumerate(syls_vocab)}
  idx2str = np.array(syls_vocab)

  alphas_start = len(special_tokens) + len(punctuation) + 1 # + 1 is for '' token
  
  # # remove rare words (threshold t)
  # threshold = 15
  # unknowns = [syl for syl in alphas if flattened.count(syl) < threshold]
  # np.delete(idx2str, np.searchsorted(idx2str, unknowns)-1)
  # str2idx.update({unk:0 for unk in unknowns})

  if verbose:
    print("\n## SPECIAL_TOKENS ##\n")
    print(sorted(special_tokens, key=len))
    print("\n## PUNCTUATION \n")
    print(sorted(punctuation, key=len))
    print("\n## PUNCTUATED_SYLS ##\n")
    print(sorted(punctuated_syls, key=len))
    print("\n## ALPHAS ##\n")
    print(sorted(alphas, key=len))
    print("\n## VOCAB ##\n")
    print(syls_vocab)
    print("\n##################\n")

  return words, alphas_start, len(syls_vocab), str2idx, idx2str
  
# # Create vocabulary and mappings from input list of lists
# def create_vocab(files, myorder, sep:str=' ', threshold:int=15, verbose:bool=False):

#   # flatten list_of_lists to list of tokens
#   flattened = []
#   for i, file_name in enumerate(files):
#     if file_name in myorder:
#       flattened.extend(flatten(split_tokens(files[file_name], sep)))
  
#   # create vocabulary
#   tokens_set = set(flattened)

#   # split alphas from non-alphas
#   non_alphas = []
#   alphas = []
#   for token in tokens_set:
#     if not token.isalpha():
#       non_alphas.append(token)
#     else:
#       alphas.append(token)

#   # split non-alphas in punctuation, 
#   # punctuated words and special tokens
#   punctuation = []
#   punctuated_syls = []
#   special_tokens = []
#   for token in non_alphas:
#     if len(token) == 1:
#       punctuation.append(token)
#     elif '<' in token:
#       special_tokens.append(token)
#     else:
#       punctuated_syls.append(token)

#   # follow with the rest of the tokens
#   alphas.extend(punctuated_syls)
#   alphas = sorted(alphas, key=len)
#   vocab = []
#   for group in [special_tokens, punctuation, alphas]:
#     vocab.extend(group)

#   # if missing, add empty string at position 0
#   if '' in vocab:
#     vocab.remove('')
#   vocab.insert(0, '')

#   # Creating a mapping from unique characters to indices
#   str2idx = {u:i for i, u in enumerate(vocab)}
#   idx2str = np.array(vocab)
  
#   # # remove rare words (threshold t)
#   # unknowns = [syl for syl in alphas if flattened.count(syl) < threshold]
#   # np.delete(idx2str, np.searchsorted(idx2str, unknowns)-1)
#   # str2idx.update({unk:0 for unk in unknowns})

#   if verbose:
#     print("\n## SPECIAL_TOKENS ##\n")
#     print(sorted(special_tokens, key=len))
#     print("\n## PUNCTUATION \n")
#     print(sorted(punctuation, key=len))
#     print("\n## PUNCTUATED_SYLS ##\n")
#     print(sorted(punctuated_syls, key=len))
#     print("\n## ALPHAS ##\n")
#     print(sorted(alphas, key=len))
#     print("\n## VOCAB ##\n")
#     print(vocab)
#     print("\n##################\n")

#   return len(vocab), str2idx, idx2str

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

def split_input_target_comedy(text_list, str2idx, inp_len=3, tar_len=4, skip=1, 
                              batch_size=64, repetitions=100):

  '''splits comedy dataset in input-target couples'''
  
  inputs = []
  targets = []

  # Prepare data for model (list of integers)
  dataset = split_tokens(text_list)
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

def split_input_target_production(datasets, str2idx, inp_len=3, tar_len=3, skip=1, 
                                  batch_size=64, repetitions=100):
  
  '''splits production dataset in input-target couples'''

  datasets_list = []
  inputs = []
  targets = []

  for text_list in datasets:

    # Prepare data for model (list of integers)
    dataset = split_tokens(text_list)
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
