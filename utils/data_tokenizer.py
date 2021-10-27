########################## SETUP ##########################
import os

# PATHS
original_path = "../data/original/"
tokenized_path = "../data/tokenized/"

# SPECIAL TOKENS 
sov = '<V>'  # start-of-verse
eov = '</V>' # end-of-verse
sot = '<T>'  # start-of-tercet
eot = '</T>' # end-of-tercet

# OUT FOLDER CREATION
if not os.path.exists(tokenized_path):
    os.mkdir(tokenized_path)
    print("CREATED: ", tokenized_path)

####################### TOKENIZATION #######################

# Files tokenization
for tokenization in [None, 'spaces']:
    print("TOKENIZATION: ", tokenization)
    for filename in os.listdir(original_path):
        if filename.endswith(".txt"):
            
            print(f"TOKENIZING FILE: {filename}")

            if not tokenization:
                sep = '<S>'
            elif tokenization == 'spaces':
                sep = '|'  

            # load text file as list of strings
            with open(os.path.join(original_path, filename), 'r') as f:
                text = f.readlines()

            # initialize variables
            verse_count = 1
            tokenized = []

            # tokenize verses
            for verse in text:
                
                # replace characters
                if not tokenization:
                    verse = verse.replace(' ', f' {sep} ')
                    verse = verse.replace('|', ' ')
                elif tokenization == 'spaces':
                    verse = verse.replace('|', sep)

                # clean remaining
                verse = verse.replace('   ', ' ')
                verse = verse.replace('  ', ' ')
                verse = verse.replace('\n', '')

                # add sov and eov tokens to verse
                if not tokenization:
                    verse = sov + verse + eov
                elif tokenization == 'spaces':
                    verse = sov + verse + sep + eov
                
                # manage tercets if needed
                if 'comedy' in filename:
                    if verse_count == 1:
                        if not tokenization:
                            verse = sot + ' ' + verse
                        if tokenization == 'spaces':
                            verse = sot + verse
                    elif verse_count == 3:
                        verse_count = 0    
                    verse_count += 1

                # special cases management
                if not tokenization:
                    if '<V> ' not in verse:
                        verse = verse.replace('<V>', '<V> ')

                # append verse to tokenized verses list
                tokenized.append(verse)

            # save tokenized text
            out_name = f'{tokenized_path}tokenized_{filename}'
            if tokenization:
                out_name = out_name.replace(".txt", f"_{tokenization}.txt")
            with open(out_name, 'w') as f:
                for verse in tokenized:
                    # print(verse)
                    f.write(verse+'\n')