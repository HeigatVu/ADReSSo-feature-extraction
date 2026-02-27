from functools import lru_cache
from typing import Counter
import spacy
import math
from ideadensity import depid

# Traditional Linguistic features
@lru_cache
def __load_spacy(lang:str="en"):
    """Support calling one time spacy model -> faster speed
    """
    if lang == "en":
        return spacy.load("en_core_web_sm")
    else:
        print("Current does not support")

def clean_and_tokenize_spacy(transcript:str, lang:str="en"):
    nlp = __load_spacy(lang)
    doc = nlp(transcript or "")
    words = []
    for word in doc:
        if word.is_alpha:
            words.append(word)

    return words, doc


def lexical_richness(transcipt:str, lang:str="en"):
    """ Meansure lexical richness
    """
    words, _ = __load_spacy(transcipt, lang)
    
    N = len(words) # Total num of word
    freqs = Counter(words) # Frequency of each word in transcript
    V = len(V) # Unique words

    # Corrected type-token radio // https://lexicalrichness.readthedocs.io/en/latest/docstring_docs.html
    if N > 0:
        cttr = round(V/math.sqrt(2*N), 2)
    else:
        cttr = 0

    # Brunet // https://arxiv.org/pdf/2109.11010
    if (N > 0) and (V > 0):
        brunet = round(N**(V**(-0.165)), 2)
    else:
        brunet = 0

    # Honore statistic // https://arxiv.org/pdf/2109.11010
    v1 = 0
    for c in freqs.values():
        if c == 1:
            v1 += 1

    honore = (100*math.log(N))/(1 - (v1/V))

    # Standardised Entropy in linguistic /‌/ https://arxiv.org/pdf/2109.11010
    if N > 1:
        entropy = 0
        for count in freqs.values():
            p_xi = count/N
            entropy -= p_xi * math.log(p_xi)
        std_entropy = round((entropy/math.log(N)), 2)
    else:
        std_entropy = 0

    # Idea density // https://aclanthology.org/K17-1033.pdf
    density,_,_ = depid(transcipt or "", is_depid_r=True)
    pidensity = round(float(density), 5)

    return cttr, brunet, honore, std_entropy, pidensity

## Part-of-speech (POS) tag



## 