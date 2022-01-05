"""
Utilitary library used in assignment 7 of the class IFT6285
University of Montréal, Autumn 2021

@authors: Maxime Monrat, Juan Duran, 2021, Nov 19
"""
import os, re, string, unicodedata

def Get_Models_Names(dir):
    """
    Returns the list of models contained in a directory
    """
    models_names = []
    if os.path.exists(dir):
        for dirpath, directories, files in os.walk(dir):
            for filename in files:
                if(filename.split('.')[-1]=='tagger'):
                    models_names.append(filename)
    else:
        print('ERROR -- Missing Directory : ' + dir + ' cannot be reached.')
    
    return models_names

def BasicFeatureFunc(token, feature_list, idx):
    """
    Basic feature extraction function, very similar to the default _get_features in the CRFTagger class, but with info on the relative position of the word
    """

    # Capitalization
    if token[0].isupper():
        feature_list.append(str(idx) + "_CAPITALIZATION")

    # Number
    if re.search(re.compile(r"\d"), token) is not None:
        feature_list.append(str(idx) + "_HAS_NUM")

    # Punctuation
    punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
    if all(unicodedata.category(x) in punc_cat for x in token):
        feature_list.append(str(idx) + "_PUNCTUATION")

    # Suffix up to length 3
    if len(token) > 1:
        feature_list.append(str(idx) + "_SUF_" + token[-1:])
    if len(token) > 2:
        feature_list.append(str(idx) + "_SUF_" + token[-2:])
    if len(token) > 3:
        feature_list.append(str(idx) + "_SUF_" + token[-3:])

    feature_list.append(str(idx) + "_WORD_" + token)

    return feature_list

def CustomFeatureFunc(tokens, idx):
    """
    Similar to the default _get_features func from the CRFTagger class, but takes context into account
    """
    token = tokens[idx]
    feature_list = []

    if not token:
        return feature_list
    
    if idx >= 1: # Check previous word, if there is one
        # if idx >= 2: # Check the second previous word, if there is one
        #     token = tokens[idx-2] 
        #     feature_list += BasicFeatureFunc(token, feature_list, -2)
        token = tokens[idx-1]
        feature_list += BasicFeatureFunc(token, feature_list, -1)
    token = tokens[idx]
    feature_list += BasicFeatureFunc(token, feature_list, 0) # Current word
    if idx < len(tokens)-1: # Check the next word, if there is one
        # if idx < len(tokens)-2: # Check the second next word, if there is one
        #     token = tokens[idx+2]
        #     feature_list += BasicFeatureFunc(token, feature_list, 2)
        token = tokens[idx+1]
        feature_list += BasicFeatureFunc(token, feature_list, 1)

    return feature_list

def word2features(tokens, i):
    """
    This feature-extracting function has been written by Aiswarya Srininvas, and can be found here:
    https://github.com/AiswaryaSrinivas/DataScienceWithPython/blob/master/CRF%20POS%20Tagging.ipynb
    """
    word = tokens[i]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = tokens[i-1]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = tokens[i-2]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
        })

    if i < len(tokens)-1:
        word1 = tokens[i+1]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['EOS'] = True
    if i < len(tokens) - 2:
        word2 = tokens[i+2]
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
        })

    return features
