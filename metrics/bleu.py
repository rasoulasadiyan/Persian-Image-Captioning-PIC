from collections import Counter
import math
from hazm import word_tokenize
import re

def tokenize(sentence):
    sentence = re.sub(r'[\u064B-\u0652]', '', sentence)
    return word_tokenize(sentence)

def ngram_counts(sentence, n):
    return Counter([' '.join(sentence[i:i+n]) for i in range(len(sentence) - n + 1)])

def clipped_ngram_counts(candidate, references, n):
    c_counts = ngram_counts(candidate, n)
    r_counts = Counter()
    for reference in references:
        r_counts.update(ngram_counts(reference, n))
    
    clipped_counts = Counter()
    for ngram, count in c_counts.items():
        clipped_counts[ngram] = min(count, r_counts[ngram])
    
    return clipped_counts

def precision(candidate, references, n):
    c_counts = ngram_counts(candidate, n)
    clipped_counts = clipped_ngram_counts(candidate, references, n)
    return sum(clipped_counts.values()) / max(1, sum(c_counts.values()))

def brevity_penalty(candidate, references):
    c_len = len(candidate)
    r_len = min((abs(len(r) - c_len), len(r)) for r in references)[1]
    if c_len > r_len:
        return 1
    else:
        return math.exp(1 - r_len / c_len)

def bleu_score(candidate, references, weights=(0.25, 0.25, 0.25, 0.25)):
  try:
    candidate = tokenize(candidate)
    references = [tokenize(ref) for ref in references]
    
    n = len(weights)
    precisions = [precision(candidate, references, i) for i in range(1, n + 1)]
    
    # Calculate weighted geometric mean of precisions
    weighted_geometric_mean = 1
    for weight, p in zip(weights, precisions):
        weighted_geometric_mean *= p ** weight
    
    bp = brevity_penalty(candidate, references)
    
    return bp * weighted_geometric_mean

  except:
    return 0.0

def compute(candidate, references):
    scores = {}
    scores['bleu-1000'] = bleu_score(candidate, references, weights=(0.25,0,0,0))

    scores['bleu-1100'] = bleu_score(candidate, references, weights=(0.25,0.25,0,0))

    scores['bleu-1110'] = bleu_score(candidate, references, weights=(0.25,0.25,0.25,0))

    scores['bleu-1111'] = bleu_score(candidate, references, weights=(0.25,0.25,0.25,0.25))

    return scores
  
