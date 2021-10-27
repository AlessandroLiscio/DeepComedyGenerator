import os
import argparse
from hyphenation.models import *

# TODO: add tercets and verses tags: <t> <v> </v>
"""
Given a folder with normal texts, generates the syllabified and cleaned text files
"""

# def add_ending_spaces(string):
#     if string[-1] != " ":
#         return string + " "
#     else: 
#         return string


# def remove_punctuation(string):
#     punct = "\".,;:!?»«-—“”()" # TODO: let's do it better :) 
#     return "".join(
#         [char for char in string if char not in punct]
#     )


# def remove_extra_spaces(string):
#     while "  " in string:
#         string = string.replace("  ", " ")
#     return string




if __name__ == "__main__":
    
    root = "data/opere"

    
    print("files: ", os.listdir(root), "(", len(os.listdir(root)), ")")

    for file_name in ["commedia.txt"]:#os.listdir(root):
        for use_punctuation in [False]:#, True]:
            for final_spaces in [True]:#, False]:
                for only_11_syl in [False]:#, False]:
                    for init_spaces in [True]:#, False]:

                        hyphenator = cnnhyphenator(from_pretrained="cnn", punctuation=use_punctuation,
                                                    starting_spaces=init_spaces, ending_spaces=final_spaces)

                        path = f"{root}/{file_name}"
                        print(f"\n\nprocessing {path}")
                        print(f"- punctuation:   {use_punctuation}")
                        print(f"- ending spaces: {final_spaces}")
                        print(f"- inital spaces: {init_spaces}")
                        print(f"- only 11-syll:  {only_11_syl}")
                        corpus = []

                        with open(path) as f:
                            text = f.read().split("\n")
                            tot = len(text)

                            for i,original_verse in enumerate(text):
                                verse = hyphenator(original_verse)                                       # hyphenate
                                n_syl = len(verse.split("|")[1:])                               # count syllables (first token is "" when splitting)
                                if only_11_syl and n_syl != 11: continue                        # keep only 11-token verses
                                
                                print("\t\r", i+1, "/", tot, end="")
                                corpus.append(verse + "\n")


                            cleaned_file_name = path.replace(".txt", 
                                                            "_clean" +
                                                            ("_np" if not use_punctuation else "") +
                                                            ("_es" if final_spaces else "") +
                                                            ("_is" if init_spaces else "") +
                                                            ("_11" if only_11_syl else "") +
                                                            ".txt")
                            
                            with open(cleaned_file_name, "w") as f:
                                f.writelines(corpus)
                                print("\n", cleaned_file_name, " saved")
print("\a")




