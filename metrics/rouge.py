from collections import Counter
import math
from hazm import word_tokenize
import re

def tokenize(sentence):
    sentence = re.sub(r'[\u064B-\u0652]', '', sentence)
    return word_tokenize(sentence)

def ngram_counts(sentence, n):
    return Counter([' '.join(sentence[i:i+n]) for i in range(len(sentence) - n + 1)])

def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    lcs.reverse()
    return lcs

def rouge_n(candidate, references, n=1):
    candidate = tokenize(candidate)
    references = [tokenize(ref) for ref in references]
    
    c_counts = ngram_counts(candidate, n)
    total_clipped_counts = Counter()
    
    for reference in references:
        r_counts = ngram_counts(reference, n)
        clipped_counts = Counter()
        for ngram, count in c_counts.items():
            clipped_counts[ngram] = min(count, r_counts[ngram])
        total_clipped_counts += clipped_counts
    
    total_matches = sum(total_clipped_counts.values())
    total_ngrams = sum(c_counts.values())
    
    precision = total_matches / total_ngrams if total_ngrams > 0 else 0
    recall = total_matches / sum(sum(r_counts.values()) for r_counts in [ngram_counts(ref, n) for ref in references]) if references else 0
    
    return {'precision':precision, 'recall':recall}

def rouge_l(candidate, references):
    candidate = tokenize(candidate)
    references = [tokenize(ref) for ref in references]
    
    total_lcs_length = 0
    total_candidate_length = 0
    
    for reference in references:
        lcs = longest_common_subsequence(candidate, reference)
        total_lcs_length += len(lcs)
        total_candidate_length += len(candidate)
    
    precision = total_lcs_length / total_candidate_length if total_candidate_length > 0 else 0
    recall = total_lcs_length / sum(len(ref) for ref in references) if references else 0
    
    return {'precision':precision, 'recall':recall}

def compute(candidate, references):
    return {
        'rouge1': rouge_n(candidate, references, n=1),
        'rouge2': rouge_n(candidate, references, n=2),
        'rougeL': rouge_l(candidate, references)
    }
    