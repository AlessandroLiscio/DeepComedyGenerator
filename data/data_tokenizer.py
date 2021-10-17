########################## SETUP ##########################

# PATHS
original_path = "original/"
tokenized_path = "tokenized/"

# SPECIAL TOKENS
sep = '<S>'  # separator
sov = '<V>'  # start-of-verse
eov = '</V>' # end-of-verse
sot = '<T>'  # start-of-tercet
eot = '</T>' # end-of-tercet

# FILES WITH TERCETS
has_tercets = ['commedia.txt']

# OUT FOLDER CREATION
if not os.path.exists(tokenized_path):
    os.mkdir(tokenized_path)
    print("CREATED: ", tokenized_path)

####################### TOKENIZATION #######################

# Files tokenization
for filename in os.listdir(original_path):
    if filename.endswith(".txt"):

        print(f"\n\nTOKENIZING FILE: {filename}\n\n")

        # load text file as list of strings
        with open(os.path.join(directory, filename), 'r') as f:
             text = f.readlines()

        # initialize variables
        verse_count = 1
        tokenized = []

        # tokenize verses
        for verse in text:

            # replace characters
            verse = verse.replace(' ', f' {sep} ')
            verse = verse.replace('|', ' ')
            verse = verse.replace('  ', ' ')
            verse = verse.replace('\n', '')

            # add sov and eov tokens to verse
            verse = sov + verse + ' ' + eov
            
            # manage tercets if needed
            if filename in has_tercets:
                if verse_count == 1:
                    verse = sot + ' ' + verse
                elif verse_count == 3:
                    verse_count = 0    
                verse_count += 1

            # special cases management
            if '<V> ' not in verse:
                verse = verse.replace('<V>', '<V> ')

            # append verse to tokenized verses list
            tokenized.append(verse)

        # save tokenized text
        with open(f'tokenized/tokenized_{filename}', 'w') as f:
            for verse in tokenized:
                # print(verse)
                f.write(verse+'\n')