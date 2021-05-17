from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import collections
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from nltk.corpus import brown
from nltk.corpus import wordnet as wn

import argparse

def preprocess_word(word:str, stopwordlist:list):
    word = word.lower()

    if word in [",", "."]:
        return None

    if "'s" in word:
        word = word.replace("'s", "")

    if word in stopwordlist:
        return None

    lemma = wn.morphy(word)
    if lemma is None:
        return word
    elif lemma in stopwordlist:
        return None
    else:
        return lemma

def preprocess_doc(document:list, stopwordlist:list):
    document = [preprocess_word(w, stopwordlist) for w in document]
    document = [w for w in document if w is not None]
    return document

def vectorize_doc(document, model):
    def vectorize_word(word, model):
        try:
            return model[word]
        except KeyError:
            #print(f'Not found: {word}')
            return None
        except Exception:
            print(Exception)
            return None

    doc_matrix = [vectorize_word(word, model) for word in document]
    doc_matrix = np.array([w for w in doc_matrix if w is not None])
    doc_vector = np.mean(doc_matrix, axis=0)
    return doc_vector

def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_cluster', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    docs = [brown.words(fileid) for fileid in brown.fileids()]
    categories = [brown.categories(fileid) for fileid in brown.fileids()]

    en_stop = nltk.corpus.stopwords.words('english')
    en_stop= ["``","/",",.",".,",";","--",":",")","(",'"','&',"'",'),',',"','-','.,','.,"','.-',"?",">","<"]                  \
        +["0","1","2","3","4","5","6","7","8","9","10","11","12","86","1986","1987","000"]                                                      \
        +["said","say","u","v","mln","ct","net","dlrs","tonne","pct","shr","nil","company","lt","share","year","billion","price"]          \
        +en_stop

    vectorizer = api.load('glove-wiki-gigaword-100')

    preprocessed_docs = [preprocess_doc(document, en_stop) for document in docs]
    vectorized_docs = [vectorize_doc(doc, vectorizer) for doc in preprocessed_docs]

    num_clusters = cli_args().num_cluster

    kmeans = KMeans(n_clusters=num_clusters, random_state=314)

    clusters = kmeans.fit_predict(vectorized_docs)
    print(clusters)

    with open(f"/workspaces/doc_analysis/output/result-with-{num_clusters}.csv", mode='w') as f1:
        with open(f"/workspaces/doc_analysis/output/origin.csv", mode='w') as f2:
            for doc, cluster, category, origin in zip(preprocessed_docs, clusters, categories, docs):
                f1.write(f'{cluster}, {category[0]}\n')
                f2.write(f'{" ".join(doc[:10])}, {" ".join(origin[:20])}\n')

if __name__ == "__main__":
    main()