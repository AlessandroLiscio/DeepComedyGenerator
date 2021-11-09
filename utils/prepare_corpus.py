import re

# PARAMETERS
lower = True
no_punct = True
space_is_token = False
use_tercets = False
add_tags = False
use_initial_spaces = True

# GLOBALS
sep = "|"
punctuation = "\".,;:!?-—“«(”»)"
prepared_corpus = []
data_path = "..\\data\\original\\divina_syll_textonly.txt"
dest_path = "corpus.txt"



def remove_punctuation(v):
    for p in punctuation:
        if p in v:
            v = v.replace(p, "").strip()
    return v.strip()


def remove_multiple_spaces(v):
    while "  " in v:
        v = v.replace("  ", " ")
    return v


def tokenize_spaces(v):
    # remove initial and final spaces
    v = v.replace(" ", "| ")
    v = re.sub(r"([|][ ])([^|])", r"\g<1>|\g<2>", v)
    return v


def insert_tags(v):
    if (len(prepared_corpus)+1) % 3 == 0 and use_tercets:
        v = v + sep + "</t>"
    else:
        v = v + sep + "</v>"
    return v


def add_initial_spaces(v):
    return " " + re.sub(r"([ ][|])([^\s])", r"\g<1> \g<2>", v)


# MAIN #

with open(data_path, encoding="utf-8") as f:
    comedy = f.read().split("\n")

    for verse in comedy:
        v = verse[1:]

        if no_punct:    
            v = remove_punctuation(v)

        if lower:           
            v = v.lower()
        
        v = remove_multiple_spaces(v)
            
        if space_is_token:
            v = tokenize_spaces(v)
        else:
            v += " "

        if use_initial_spaces:
            v = add_initial_spaces(v)

        if add_tags:
            v = insert_tags(v)

        if v[0] == sep:
            v = v[1:]

        if v[0] == " " and v[1] == sep:
            v = " "+v[2:]

        prepared_corpus.append(v)


# ANALYSIS
verses_lenght = {}
vocab = set()
tot_verses = len(prepared_corpus)

with open(dest_path, "w", encoding="utf-8") as f:
    for verse in prepared_corpus:
        f.write(verse + "\n")
        syllables = verse.split("|")
        n_syll = len(syllables)
        if n_syll in verses_lenght:
            verses_lenght[n_syll][0] += 1
        else:
            verses_lenght[n_syll] = [1, verse]

        vocab.update(set(syllables))


print("\n\n")
for n in verses_lenght:
    print("{:>3}-syll verses:\t {:>6} ({:>7.3f} %)\t sample: {}".format(n, verses_lenght[n][0], round(verses_lenght[n][0]/tot_verses*100, 3), [verses_lenght[n][1]]))

print("\n vocab lenght: ", len(vocab))
