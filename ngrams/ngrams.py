import string

import torch

def generate_ngrams(sentence, n=(1, 2)):
    table = str.maketrans("", "", string.punctuation)
    words = [w.translate(table) for w in sentence.lower().split()]
    words = [w for w in words if w]
    ngrams = []
    for size in n:
        for i in range(len(words) - size + 1):
            ngrams.append(" ".join(words[i:i + size]))
    return ngrams

def sentence_to_ngram_bow(sentence, ngram_to_idx, n=(1, 2)):
    vec = torch.zeros(len(ngram_to_idx))
    for ng in generate_ngrams(sentence, n):
        if ng in ngram_to_idx:
            vec[ngram_to_idx[ng]] = 1
    return vec