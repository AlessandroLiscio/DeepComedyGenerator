########################## SETUP ##########################
import os

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
has_tercets = ['commedia.txt', 'commedia_squares.txt', 'commedia_quotesless_squares.txt']

# REMOVE PUNCTUATION (DATASET CHARS ONLY)
rm_quotes = False
rm_punctuation = True
assert not rm_quotes == rm_punctuation == True

if rm_quotes:
    print("REMOVING QUOTES")
elif rm_punctuation:
    print("REMOVING PUNCTUATION")

punctuation   = '-:,?“\)—»«!”\(";.' # no apostrophe here
# punctuation = '-:,?“‘\)—»«!”\(";.’'
quotes = "»«“”\""

# OUT FOLDER CREATION
if not os.path.exists(tokenized_path):
    os.mkdir(tokenized_path)
    print("CREATED: ", tokenized_path)

####################### TOKENIZATION #######################

# Files tokenization
for filename in os.listdir(original_path):
    if filename.endswith(".txt"):

        print(f"TOKENIZING FILE: {filename}")

        # load text file as list of strings
        with open(os.path.join(original_path, filename), 'r') as f:
             text = f.readlines()

        # initialize variables
        verse_count = 1
        tokenized = []

        # tokenize verses
        for verse in text:

            if len(verse.split("|")) -1  == 11:

                # replace characters
                verse = verse.replace('|', sep)

                # remove punctuations
                if rm_punctuation:
                    for char in verse:
                        if char in punctuation:
                            verse = verse.replace(char, '')
                elif rm_quotes:
                    verse = verse.replace('»', '')
                    verse = verse.replace('«', '')
                    verse = verse.replace('“', '')
                    verse = verse.replace('”', '')
                    verse = verse.replace('"', '')

                # clean remaining
                verse = verse.replace('   ', ' ')
                verse = verse.replace('  ', ' ')
                verse = verse.replace('\n', '')

                # add final spaces for syllables coherence
                if verse[-1] not in (punctuation+" "):
                    verse += " "

                # add sov and eov tokens to verse
                verse = sov + verse + sep + eov
                
                # manage tercets if needed
                if filename in has_tercets:
                    if verse_count == 1:
                        verse = sot + verse
                    elif verse_count == 3:
                        verse_count = 0    
                    verse_count += 1

                # append verse to tokenized verses list
                tokenized.append(verse)

        # save tokenized text
        out_name = f'tokenized/tokenized_{filename}'
        if rm_quotes: out_name = out_name.replace('.txt','_quotesless.txt')
        elif rm_punctuation: out_name = out_name.replace('.txt', '_punctuationless.txt')
        with open(out_name.replace(".txt", "_spaces_eleven.txt"), 'w') as f:
            for verse in tokenized:
                # print(verse)
                f.write(verse+'\n')