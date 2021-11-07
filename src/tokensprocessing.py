import numpy as np

# Replaces special tokens with more readable ones and remove weird spaces
def clear_text(text):
  if text[0] == " ":
    text = text[1:]

  text = text.replace('~', ' ')
  text = text.replace('<s>', ' ')

  text = text.replace('  ', ' ')

  text = text.replace('<v>', '')
  text = text.replace('</v>', '\n')
<<<<<<< HEAD
  text = text.replace('</t>', '\n\n')
  text = text.replace('~', ' ')
=======

  text = text.replace('<t>','\n')
  text = text.replace('</t>','\n\n')
  
>>>>>>> beam_search
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

# Converts our text values to numeric.
def text_list_to_int(string_list, str2idx):
  return np.array([str2idx[c] for c in string_list])
 
# Converts our numeric values to text.
def ints_to_text(ints, idx2str):
  try:    ints = ints.numpy()
  except: pass
  return ''.join(idx2str[ints])
