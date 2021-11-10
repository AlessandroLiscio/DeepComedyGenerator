import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import DataLoader
import json

verbose = False

log = False
in_path = '../data/tokenized/sov/' # "verses" or "tercets" does not matter
out_path = '../data/vocabs/'

if not os.path.exists(out_path):
    os.mkdir(out_path)
    print("CREATED: ", out_path)

print("GENERATING VOCABULARIES:")
for comedy_name in os.listdir(in_path):

  comedy_name = comedy_name.replace("tokenized_","")
  comedy_name = comedy_name.replace(".txt","")


  if "base" in comedy_name:
    tokenization = "base"
    comedy_name = comedy_name.replace("_base","")
  elif "_is-es" in comedy_name:
    tokenization = "is-es"
    comedy_name = comedy_name.replace("_is-es","")
  elif "_es" in comedy_name:
    tokenization = "es"
    comedy_name = comedy_name.replace("_es","")
  elif "_is" in comedy_name:
    tokenization = "is"
    comedy_name = comedy_name.replace("_is","")

  dataloader = DataLoader(in_path=in_path,
                          comedy_name=comedy_name,
                          tokenization=tokenization,
                          inp_len=3,
                          tar_len=4,
                          repetitions_production=0,
                          repetitions_comedy=1,
                          verbose = verbose)

  out_dict = dict(enumerate(dataloader.vocab))
  if log:
    out_dict['log'] = dataloader.vocab_info
  
  print(" - {:<30}{}".format(f"{comedy_name}_{tokenization}:", dataloader.vocab_info))

  json.dump(out_dict, open(f'{out_path}vocab_{comedy_name}_{tokenization}.json', 'w'))