import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_folder(path, verbose:bool=True):
    if not os.path.exists(path):
        os.mkdir(path)
        if verbose: print("CREATED: ", path)

in_path = "../data/hyphenated/"
data_path = "../data/tokenized/"
create_folder(data_path)

for dataset in ['tercets', 'tercets_sov', 'tercets_sot', 'tercets_sov_sot', 'verses', 'verses_sov']:

    out_path = data_path+f"{dataset}/"
    create_folder(out_path, verbose=True)

    for filename in os.listdir(in_path):

        new_text = []
        count = 1

        with open(in_path+filename, "r") as f:
            text = f.readlines()

        for row in text:

            if "sov" in dataset:
                row = '<v>|'+row

            row = row.replace('\n','|</v>\n')

            if "tercets" in dataset:
                if count == 3:
                    row = row.replace('</v>', '</t>')
                    count = 0
                if "sot" in dataset and count == 1:
                    if "sov" in dataset:
                        row = row.replace('<v>', '<t>')
                    else:
                        row = '<t>'+row
                count += 1

            new_text.append(row)

        filename = filename.replace("hyphenated", "tokenized")
        with open(out_path+filename, "w") as f:
            for row in new_text:
                f.write(row)
            