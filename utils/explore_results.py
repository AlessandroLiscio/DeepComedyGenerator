import pandas as pd

results_table_file = "table_16-11.csv"#"table_15-11_results.csv"
print("\n\n Exploring ", results_table_file, "\n\n")
df = pd.read_csv(results_table_file, sep=";")
table = df.drop(columns="path")                                 # remove path

# # take the bests
# table = table.sort_values(by=["hendec"], ascending=False)       # sort by hendec score
# table = table[table["hendec"] > 0.95]                           # select good hendec scores
# table = table[table["tot_words"] > 50]                          # select texts with lot of words
# table = table[table["plagiarism"] < 0.75]                       # select text with few plagiarism

# take only the beam search
table = table[table["gen_mode"] == "beam_search"]
table = table.sort_values(by=["hendec"], ascending=False)       # sort by hendec score
table = table[table["hendec"] > 0.95]                           # select good hendec scores
# table = table[table["tot_words"] > 50]                          # select texts with lot of words
# table = table[table["plagiarism"] < 0.75]                       # select text with few plagiarism

print(table)

id = int(input("select an id: "))
print("\n\n")
path = df.iloc[id]["path"]




# OPEN SELECTED FILE

with open(path, encoding="utf-8") as f:
    text = f.read().split("\n")
    print("file: ", path, "\n")

for line in text:
    print(line)

txt_filename = path.split("/")[-1]
log_path = "/".join(path.split("/")[:-1]) + "/logs/" + txt_filename.replace(".txt", ".log")

with open(log_path) as f:
    print(f.read())

print("----------")
print(df.iloc[id][:-1])






