import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import DataLoader
import json

epochs_production = 0
epochs_comedy = 20
verbose = False

log = False
hyphenated_path = '../data/hyphenated/'
in_path = '../data/tokenized/'
out_path = '../data/vocabs/'

if not os.path.exists(out_path):
    os.mkdir(out_path)
    print("CREATED: ", out_path)

print("GENERATING VOCABULARIES:")
for tokenization in ['base', 'spaces']:
  for comedy_name in os.listdir(hyphenated_path):
    if not ("_is_" in comedy_name and tokenization == 'base'):

      comedy_name = comedy_name.replace(".txt","")

      dataloader = DataLoader(in_path=in_path,
                              comedy_name=comedy_name,
                              tokenization=tokenization,
                              repetitions_production=epochs_production,
                              repetitions_comedy=epochs_comedy,
                              verbose = verbose)

      out_dict = dict(enumerate(dataloader.vocab))
      if log:
        out_dict['log'] = dataloader.vocab_info
      
      print(" - {:<30}{}".format(f"{comedy_name}_{tokenization}:", dataloader.vocab_info))

      json.dump(out_dict, open(f'{out_path}vocab_{comedy_name}_{tokenization}.json', 'w'))