from cleaning import *


def plagiarism_score(text, reference, n_min=3, verbose=False):

    text = text.lower()
    text = all_in_one_line(text)
    text = remove_punctuation(text)

    reference = reference.lower()
    reference = all_in_one_line(reference)
    reference = remove_punctuation(reference)

    gen_seq = text.split(" ")

    score = 0
    i = 0
    while i <= len(gen_seq)-n_min:

        n = len(gen_seq)-i

        while n >= n_min:
            seq = gen_seq[i:i+n]
            #print(i, i+n, "  \t", " ".join(seq), " \t\tin\t", original, end="\t")
            text = " ".join(seq) 
            if text in reference:
                print("\n >", text, "\nngram dimension: ", n)
                i += n-1
                score += n
                break
            else: 
                # print()
                n -= 1
        i += 1
        
    if verbose: print(f"---------------\nscore: {score} / {len(gen_seq)}\n{round(score / len(gen_seq), 2)}")
    return round(score / len(gen_seq), 2)







if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--generation", type=str, required=False, help="generation file")
    parser.add_argument("--reference", type=str, required=False, help="original file")
    args = parser.parse_args()

    
    if args.generation == args.reference == None:
        # test
        gen = "a b c d x y d e f x y a b"
        original = "a b c d e f g h i j k l m n o"     
        print(plagiarism_score( gen,
                                original, # to use a file ---> open("data/divina_textonly.txt", encoding="utf-8").read(),
                                n_min = 3,
                                verbose=True
        ))

    else:
        print(plagiarism_score( open(args.generation, encoding="utf-8").read(),
                                open(args.reference, encoding="utf-8").read(),
                                n_min = 3,
                                verbose=True
        ))
        print("\a----------------\ncompleted.")