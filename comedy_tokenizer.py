with open('data/divina_syll_textonly.txt', 'r') as f:
    dc = f.readlines()

sep = '<S>'
sov = '<V>'
eov = '</V>'
sot = '<T>'
# eot = '</T>'

t_count = 1

tok_dc = []

for verse in dc:

    verse = verse.replace(' ', f' {sep} ')
    verse = verse.replace('|', ' ')
    verse = verse.replace('  ', ' ')
    verse = verse.replace('\n', '')

    verse = sov + verse + ' ' + eov
    
    if t_count == 1:
        verse = sot + ' ' + verse
    elif t_count == 3:
        t_count = 0
        
    t_count += 1

    if '<V> ' not in verse:
        verse = verse.replace('<V>', '<V> ')

    tok_dc.append(verse)

with open('data/tokenized/tokenized_commedia.txt', 'w') as f:
    for verse in tok_dc:
        print(verse)
        f.write(verse+'\n')