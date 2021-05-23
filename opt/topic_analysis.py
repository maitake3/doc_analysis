import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import collections
import gensim
import gensim.downloader as api
from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pyLDAvis.gensim_models

from nltk.corpus import brown
from nltk.corpus import wordnet as wn

import argparse
import json
import datetime
import clustering


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_topics', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    docs = [brown.words(fileid) for fileid in brown.fileids()]
    categories = [brown.categories(fileid) for fileid in brown.fileids()]

    #docs = docs[:50]

    en_stop = nltk.corpus.stopwords.words('english')
    en_stop= ["``", "/", ",.", ".,", ";", "--", ":", ")", "(", '"', '&', "'", '),', ',"', '-', '.,', '.,"', '.-', "?", ">", "<", "''","!"]                  \
        +["0","1","2","3","4","5","6","7","8","9","10","11","12","86","1986","1987","000"]                                                      \
        +["said","say","u","v","mln","ct","net","dlrs","tonne","pct","shr","nil","company","lt","share","year","billion","price", "would", "may", "could"]          \
        +en_stop
    preprocessed_docs = [clustering.preprocess_doc(document, en_stop) for document in docs]
    #print(" ".join(docs[0][:100]))
    #print(preprocessed_docs[0][:100])
    num_topics = cli_args().num_topics

    dictionary = corpora.Dictionary(preprocessed_docs)
    our_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    #with open('/workspaces/doc_analysis/output/token_id.json', mode='w') as f:
        #json.dump(dictionary.token2id, f, indent=4)

    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus=our_corpus,
        num_topics=num_topics,
        id2word=dictionary,
        alpha=0.1,
        eta=0.1,
        minimum_probability=0.01,
        random_state=314
    )

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    lda_display = pyLDAvis.gensim_models.prepare(ldamodel, our_corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, f'/workspaces/doc_analysis/output/{now}_lda.html')
    pyLDAvis.show(lda_display)


if __name__ == "__main__":
    main()