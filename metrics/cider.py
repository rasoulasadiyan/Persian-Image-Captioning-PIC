import numpy as np
from hazm import Normalizer, WordTokenizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

normalizer = Normalizer()
tokenizer = WordTokenizer()

def tokenize_caption(caption):
    normalized_caption = normalizer.normalize(caption)
    return tokenizer.tokenize(normalized_caption)

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def compute_tf_idf(captions, n):
    ngram_captions = []
    for caption in captions:
        tokens = tokenize_caption(caption)
        ngram_tuples = generate_ngrams(tokens, n)
        ngram_strings = [' '.join(ngram) for ngram in ngram_tuples]
        ngram_captions.append(' '.join(ngram_strings))
    
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_caption, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(ngram_captions)
    return tfidf_matrix

def compute_consensus_matrix(refs, gens, n):
    combined_captions = refs + gens
    tfidf_matrix = compute_tf_idf(combined_captions, n)
    
    ref_tfidf = tfidf_matrix[:len(refs)].toarray()
    gen_tfidf = tfidf_matrix[len(refs):].toarray()
    
    ref_tfidf = ref_tfidf / np.linalg.norm(ref_tfidf, axis=1, keepdims=True)
    gen_tfidf = gen_tfidf / np.linalg.norm(gen_tfidf, axis=1, keepdims=True)
    
    consensus_matrix = np.dot(gen_tfidf, ref_tfidf.T)
    return consensus_matrix

def cider_score(consensus_matrix):
    consensus_mean = np.mean(consensus_matrix, axis=1)
    cider_score = stats.gmean(consensus_mean)
    return cider_score

def compute(ref, gen, n=4):
  try:
    refs = [ref]
    gens = [gen]
    consensus_matrix = compute_consensus_matrix(refs, gens, n)
    return cider_score(consensus_matrix)

  except:
    return 0.0
