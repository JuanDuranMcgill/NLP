import re
import bz2
import random
from progress.bar import Bar

def TextCleanup(text):
    """
    Splits a text in a vector of words, with preprocessing:
    - lowercasing
    - removes special characters
    - removes noisy text of the shape "<NUM>something"
    - replaces numbers by <NUM>
    - replaces URLs and email adresses by <URL>
    - replaces laughs by <LAUGHS>
        Input: the text, as a string
        Output: the text    
    """

    # Lowercase the text
    text = text.lower()
    # Replace URLs and email adresses
    text=re.sub("(https?:\\/\\/(www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_\\+.~#?&//=]*))|((www\\.)[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_\\+.~#?&//=]*))|(www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\/([-a-zA-Z0-9()@:%_\\+.~#?&//=]+)|(\\S?)+\\@\\S+\\.\\w+","<URL>",text)
    # Replace laughs
    text=re.sub("[aA]?([Hh][aeiAEI]){2,}[hH]?", "<LAUGHS>", text)
    # Remove noisy words only composed of voyels
    text=re.sub("[aeiouyAEIOUY]{5,}", " ", text)
    # Remove noisy text of the shape "<NUM>something"
    text=re.sub("[0-9]+\\S+"," ",text)
    # Replace numbers
    text=re.sub("[0-9]+","<NUM>",text)
    # Remove special characters
    #text=re.sub("\\W"," ",text)

    return text


random.seed(0)

# with open('1BW_tok', 'rt') as fin, open('1BW.ref', 'w', encoding='utf-8') as fref, open('1BW.train', 'w', encoding='utf-8') as ftrain, Bar("Reading the corpus...", max = 30301028, suffix='%(percent)d%% - %(eta)ds') as progressBar:
#     for line in fin:        
#         toks = line.split()
#         if len(toks) <= 25 and len(toks) >= 5:
#             fref.write(line)            
#             random.shuffle(toks)
#             ftrain.write(' '.join(toks) + '\n')
#         progressBar.next()