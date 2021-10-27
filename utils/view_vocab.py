import os
import json


print("MODEL")
with open("vocab_ric.json", "r", encoding="utf-8") as f:
    tokens_spaces = set(json.load(f).values())

ending_word_tokens_1 = set()
synalephe_tokens_1 = set()
normal_tokens_1 = set()

for token in tokens_spaces:
    if token == "": continue
    
    # ending word tokens
    if token[-1] == " ":
        ending_word_tokens_1.add(token)
    
    # synalephe tokens
    if token.count(" ") > 0 and token[-1] != " ":
        synalephe_tokens_1.add(token)

    if " " not in token:
        normal_tokens_1.add(token)

print("vocab tokens:            ", len(tokens_spaces))
print("ending word tokens:      ", len(ending_word_tokens_1))
print("synalephe tokens:        ", len(synalephe_tokens_1))
print("normal tokens:           ", len(normal_tokens_1))



print("\n\nRECOMPUTED")
vocab = set()
with open("data/opere/commedia_clean_np_es_11.txt", encoding="utf-8") as f:
    for verse in f.read().split("\n"):
        tokens = []
        for token in verse.split("|")[1:]:
            while "  " in token:
                token = token.replace("  ", " ")
            tokens.append(token)
        vocab = vocab.union(tokens)

ending_word_tokens_2 = set()
synalephe_tokens_2 = set()
normal_tokens_2 = set()

for token in vocab:
    if token == "": continue
    # ending word tokens
    if token[-1] == " ":
        ending_word_tokens_2.add(token)
    
    # synalephe tokens
    if token.count(" ") > 0 and token[-1] != " ":
        synalephe_tokens_2.add(token)

    if " " not in token:
        normal_tokens_2.add(token)


print("vocab tokens:            ", len(vocab))
print("ending word tokens:      ", len(ending_word_tokens_2))
print("synalephe tokens:        ", len(synalephe_tokens_2))
print("normal tokens:           ", len(normal_tokens_2))