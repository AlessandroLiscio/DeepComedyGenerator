import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import DataLoader
import json

verbose = False
log = False

tok_path = '../data/tokenized/'
voc_path = '../data/vocabs/'

if not os.path.exists(voc_path):
  os.mkdir(voc_path)
  print("CREATED: ", voc_path)

for dataset in os.listdir(tok_path):

  dataset = dataset.split("/")[-1]

  in_path = tok_path + dataset
  out_path = voc_path + dataset

  if os.path.isdir(in_path):

    print('\n', dataset.upper())

    if not os.path.exists(out_path):
      os.mkdir(out_path)
      print("CREATED: ", out_path)

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

      dataloader = DataLoader(
        from_pickle = False,
        dataloader_path = None,
        data_path = in_path,
        dataset = dataset,
        comedy_name = comedy_name,
        tokenization = tokenization,
        inp_len = 1,
        tar_len = 2,
        repetitions_production = 0,
        repetitions_comedy = 1,
        padding = 'pre',
        verbose = verbose,
        )

      out_dict = dict(enumerate(dataloader.vocab))
      if log:
        out_dict['log'] = dataloader.vocab_info
      
      print(" - {:<30}{}".format(f"{comedy_name}_{tokenization}:", dataloader.vocab_info))

      json.dump(out_dict, open(f'{out_path}/vocab_{comedy_name}_{tokenization}.json', 'w'))