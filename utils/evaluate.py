import json
import os
import argparse

from hyphenation.models import *


class Evaluator:
        
    def __init__(self, dictionary_path, original_comedy_path):
        self.dictionary = json.load(open(dictionary_path))
        
        with open(original_comedy_path, encoding="utf-8") as f:
            self.original_reference = f.read()

        self.sep = "|"
        self.accented_vowels = "àèìòù" # TODO to be replaced with regex in preprocessing script
        self.vowels = "aeiou"

    def remove_multiple_spaces(self, text):
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    def remove_punctuation(self, text):
        punct = "\"/<>()[]{}'.,;:!?»«-—“”’"
        for p in punct:
            text = text.replace(p, "")

        text = self.remove_multiple_spaces(text)
        return text

    def all_in_one_line(self, text):
        text = text.replace("\n", " ") 
        text = self.remove_multiple_spaces(text)
        return text 

    def get_stress(self, sample_syll, return_mask=False):

        punct = "\"'.,;:!?»«-—“”’" # TODO: let's do it better :) 
        sample_syll = "".join([s for s in sample_syll.lower() if s not in punct]).strip()              
        tokens = [s.strip() for s in sample_syll.split(self.sep)[1:]]                        
        last_word_start = [i for i,c in enumerate(sample_syll) if c == " "][-1] + 1
        last_word_syll = sample_syll[last_word_start:]
        last_word = last_word_syll.replace(self.sep, "")
        
        try: 
            stress_pos = len(tokens) + self.dictionary[last_word]["stress_pos"]    # get the position from the vocabulary
        except: 
            for v in self.vowels:
                try: 
                    stress_pos = len(tokens) + self.dictionary[v + last_word]["stress_pos"] # probably the word's start is trucated, try possible starting vowels 
                    break
                except:
                    stress_pos = len(tokens) - 1 # suppose the stress on the penultimo

        if return_mask:
            return stress_pos, [0 if i+1 != stress_pos else 1 for i in range(len(tokens))]
        else:
            return stress_pos


    def is_hendecasyllabic(self, hyphenated_verse, verbose=False):
       
        stress_pos, mask = self.get_stress(hyphenated_verse, return_mask=True)
        
        if verbose: 
            # Surround stressed syllable with "[ ]"
            tokens = hyphenated_verse.split(self.sep)
            stressed_verse = self.sep.join(tokens[:stress_pos] + ["["+tokens[stress_pos]+"]"] + tokens[stress_pos+1:])
            print("\n ", stressed_verse, len(mask), mask)

        if len(mask) < 10: return False
        if mask[9] != 1: return False
        return True


    def exists_in_comedy(self, word):
        return word in self.dictionary



    def incorret_words(self, verse):
        bad_words = []
        for word in verse.split(" "):
            if not self.exists_in_comedy(word):
                bad_words += [word]
        return bad_words



    def plagiarism_score(self, text, n_min=3):
        text = text.lower()
        text = self.all_in_one_line(text)
        text = self.remove_punctuation(text)
        reference = self.original_reference.lower()
        reference = self.all_in_one_line(reference)
        reference = self.remove_punctuation(reference)

        text_seq = text.split(" ")
        score = 0

        i = 0
        while i <= len(text_seq)-n_min:
            n = len(text_seq)-i
            while n >= n_min:
                seq = text_seq[i:i+n]
                if " ".join(seq) in reference:
                    i += n-1
                    score += n
                    break
                else: 
                    n -= 1
            i += 1
            
        return round(score / len(text_seq), 2)
            
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--folder", type=str, required=True, help="folder containing the generation files to be evaluated")
    args = parser.parse_args()
    
    
    results = {}
    counter = 0
    do_hyphenation = True
    
    # Initialize Verse Evaluator
    VE = Evaluator("data/hyphen_dict.json", original_comedy_path="data/divina_textonly.txt")

    if do_hyphenation:
        hyphenator = cnnhyphenator(from_pretrained="cnn-lower")
        hyphen_test = hyphenator("prova"); print("model loaded","\n"*4)     # empty first execution of the hyphenator to hide warnings


    

    root_dir = args.folder.replace('\\', "/") + ("/" if args.folder[-1] != "/" else "")
    results_table_path = f"{root_dir}table_*.csv".replace("*", root_dir.split("/")[0])

    print("Storing results in -> ", results_table_path)
    input()
    print_header = True

    for epochs in [name for name in os.listdir(root_dir) if os.path.isdir(root_dir+name)]: # epoch folder: 0_10, 0_20, ...
        
        model_info = json.load(open(root_dir+epochs +"/log.json"))
        tokenization_mode = model_info["dataloader"]["tokenization"]
        
        try:    loss_mode = f'{model_info["trainings"]["info"]["weight_eov"]}-{model_info["trainings"]["info"]["weight_sot"]}'
        except: loss_mode = "normal"

        try:    tercet_mode = model_info["dataloader"]["dataset"]
        except: tercet_mode = "?"

        path = root_dir+epochs + "/generations/" 



        for mode in [name for name in os.listdir(path) if os.path.isdir(path+name)]: # mode: "sampling"/"beam search"
            
            gen_dir = path + mode + "/"
            log_dir = gen_dir + "logs/"

            # Create logs directory if it does not exist
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            generation_files = [name for name in os.listdir(gen_dir)[:None] if name[:3] == "GEN" and name[-4:] == ".txt"]

            for n,filename in enumerate(generation_files):
                counter += 1
                temperature = filename.split("_")[1].replace(".txt", "")
                
                with open(gen_dir+filename, encoding="utf-8") as f:
                    text = f.read()

                verses = text.replace("\n\n", "\n").split("\n")
                
                n_verses = len(verses)
                matches = 0
                bad_words = []
                tot_words = set()
                verses_lenghts = {}
                log = ""
                hyphenated_verses = []

                for i, verse in enumerate(verses):

                    # print("\r{:<10}{:<15}{:<10}{:<15}{:<15}{:<15}".format(epochs, mode, temperature, tokenization_mode, tercet_mode, loss_mode), end="")
                    # print(f"\r{i}/{len(verses)}", end="    ")
                    verse = verse.replace("[", "").replace("]", "").lower()
                    punct = "\".,;:!?»«-—“”" # TODO
                    verse = "".join([c for c in verse if c not in punct]).strip()
                    if verse.replace("\n", "").strip() == "":
                        n_verses -= 1 
                        continue


                    # hyphenation
                    if do_hyphenation:
                        hyphenated_verse = hyphenator(verse)
                    else:   
                        hyphenated_verse = verse
                        if hyphenated_verse[0] != "|": hyphenated_verse = "|" + hyphenated_verse

                    hyphenated_verses.append(hyphenated_verse) # used later for rhyme checking, in order not to repeat the hyphenation


                    #  word correctness
                    tot_words.update(set(verse.split(" ")))
                    bad_words += VE.incorret_words(verse)


                    # check if is hendecasyllable 
                    if VE.is_hendecasyllabic(hyphenated_verse): 
                        matches += 1
                    else:
                        log += "\n" + hyphenated_verse


                    # verses lenght histogram  
                    tokens = hyphenated_verse.split(VE.sep)[1:] # first element is "" when splitting
                    if len(tokens) in verses_lenghts:
                        verses_lenghts[len(tokens)] += 1
                    else:
                        verses_lenghts[len(tokens)] = 1



                # word correctness
                word_correctness = round(1 - len(bad_words) / len(tot_words), 2)


                # verses lenght
                verses_lenghts = {key:round(verses_lenghts[key]/n_verses, 5) for key in sorted(verses_lenghts.keys())}


                # hencasyllabic score
                hendec_score = round(matches/n_verses, 2)


                # plagiarism score
                plag_score = VE.plagiarism_score(text, n_min=5) if hendec_score >= 0.95 else -1 # evaluate only good candidates

                results[counter] = {
                    "epochs":epochs,
                    "gen_mode": mode,
                    "temp": temperature,
                    "tokens": tokenization_mode,
                    "tercets": tercet_mode,
                    "loss_type": loss_mode,

                    "tot_words": len(tot_words),
                    "word_corr": word_correctness,
                    "hendec": hendec_score,
                    "plagiarism": plag_score,
                    "lengths": str(verses_lenghts),
                    "path": gen_dir+filename,
                    #"rhymes_ratio": VE.rhyme_ratio(hyphenated_verses)
                }



                # print table
                if print_header:
                    for key in results[counter].keys():
                        print("{:<12}".format(key), end="")
                    print("\n", "="*130)
                    print_header = False
            
                for val in results[counter].values():
                    print("{:<12}".format(val), end="")
            
                print("<---" if results[counter]["hendec"] > 0.95 else "")



                # Save local log
                with open(log_dir+filename.replace(".txt", ".log"), 'w+') as log_file:

                    log += f"\n---------\nverses lenghts:\n"
                    for key in verses_lenghts:
                        log += f"\n  {key}: {verses_lenghts[key]}"

                    log += f"\n---------\nincorrect words"
                    for w in bad_words:
                        log += f"\n  {w}"
                    
                    log_file.write(log)


                # create results summary file 
                if not os.path.exists(results_table_path):
                    with open(results_table_path, "w") as f:
                        columns = ";".join([str(col) for col in results[counter].keys()])
                        f.write(columns+"\n")
            
                # append current scores
                with open(results_table_path, "a") as f:
                    vals = ";".join([str(val) for val in results[counter].values()])
                    f.write(vals + "\n")


        # except:
        #     print("err")


print("\a\n\n-----------------------\nEvalutaion completed.")


    