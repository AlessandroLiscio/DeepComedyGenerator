from utils.dataloader import DataLoader
import json
import os

log = False
tokenized_path = 'data/tokenized/'
vocabs_folder = 'data/vocabs/'

if not os.path.exists(vocabs_folder):
    os.mkdir(vocabs_folder)
    print("CREATED: ", vocabs_folder)

for comedy_name in ['comedy', 'comedy_11']:
  for tokenization in [None, 'spaces']:

    if not tokenization:
      sep = ' '
    else:
      sep = '|'

    dataloader = DataLoader(sep = sep,
                            in_path=tokenized_path,
                            comedy_name=comedy_name,
                            tokenization=tokenization,
                            epochs_production=0,
                            epochs_comedy=50,
                            verbose = True)

    out_dict = dict(enumerate(dataloader.vocab))
    if log:
      out_dict['log'] = dataloader.vocab_info
    
    if not tokenization:
      json.dump(out_dict, open(f'{vocabs_folder}vocab_{comedy_name}.json', 'w'))
    else:
      json.dump(out_dict, open(f'{vocabs_folder}vocab_{comedy_name}_{tokenization}.json', 'w'))