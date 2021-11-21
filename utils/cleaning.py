
def remove_multiple_spaces(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text

def remove_punctuation(text):
    punct = "\"/<>()[]{}'.,;:!?»«-—“”’"
    for p in punct:
        text = text.replace(p, "")

    text = remove_multiple_spaces(text)
    return text

def all_in_one_line(text):
    text = text.replace("\n", " ") 
    text = remove_multiple_spaces(text)
    return text 
