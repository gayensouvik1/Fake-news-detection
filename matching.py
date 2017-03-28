from sklearn.feature_extraction.text import TfidfVectorizer
import os	
import codecs
import numpy as np
import itertools
import glob
import sys

documents=list(open("a"))
#documents = [open(f) for f in text_files]
tfidf = TfidfVectorizer().fit_transform(documents)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = tfidf * tfidf.T

print (pairwise_similarity)