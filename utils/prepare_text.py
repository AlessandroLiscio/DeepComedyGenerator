import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    in_dir = "../data/original"
    out_dir = "../data/hyphenated"

    # Create output folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print("CREATED: ", out_dir)

    print("files: ", os.listdir(in_dir), "(", len(os.listdir(in_dir)), ")")

    for in_file_name in os.listdir(in_dir):
        for use_punctuation in [True, False]:
            for final_spaces in [True, False]:
                for only_11_syl in [True, False]:
                    for init_spaces in [True, False]:

                        in_file_path = f"{in_dir}/{in_file_name}"
                        print(f"\n\nprocessing {in_file_path}")
                        print(f"- punctuation:   {use_punctuation}")
                        print(f"- ending spaces: {final_spaces}")
                        print(f"- inital spaces: {init_spaces}")
                        print(f"- only 11-syll:  {only_11_syl}")

                        hyphenator = cnnhyphenator(from_pretrained="cnn", punctuation=use_punctuation,
                                                    starting_spaces=init_spaces, ending_spaces=final_spaces)

                        # Read input file
                        with open(in_file_path) as f:
                            text = f.read().split("\n")
                            tot = len(text)

                        # Hyphenate verses and select verses with only 11 syllables if needed
                        corpus = []
                        for i,original_verse in enumerate(text):
                            verse = hyphenator(original_verse)                                       # hyphenate
                            n_syl = len(verse.split("|")[1:])                               # count syllables (first token is "" when splitting)
                            if only_11_syl and n_syl != 11: continue                        # keep only 11-token verses

                            print("\t\r", i+1, "/", tot, end="")
                            corpus.append(verse + "\n")


                        out_file_name = in_file_path.replace(".txt",
                                                            ("_np" if not use_punctuation else "") +
                                                            ("_es" if final_spaces else "") +
                                                            ("_is" if init_spaces else "") +
                                                            ("_11" if only_11_syl else "") +
                                                            ".txt")

                        out_file_path = f"{out_dir}/{out_file_name}"
                        
                        with open(out_file_path, "w") as f:
                            f.writelines(corpus)
                            print("\n", out_file_path, " saved")
print("\a")




