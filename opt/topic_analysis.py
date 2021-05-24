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
import random
import os
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
    #print(len(docs))

    en_stop = nltk.corpus.stopwords.words('english')
    en_stop= ["``", "/", ",.", ".,", ";", "--", ":", ")", "(", '"', '&', "'", '),', ',"', '-', '.,', '.,"', '.-', "?", ">", "<", "''","!"]                  \
        +["0","1","2","3","4","5","6","7","8","9","10","11","12","86","1986","1987","000"]                                                      \
        +["said","say","u","v","mln","ct","net","dlrs","tonne","pct","shr","nil","company","lt","share","year","billion","price"]          \
        + ["would", "may", "could", "one", "two", "three", "make", "like", "go", "well", "much", "know", "use", "us", "take", "come", "get", "see", "give", "look", "many", "must", "also", "even", "seem", "need", "show", "little", "new", "old", "might", "want", "mr", "mrs", "first", "time", "day", "tell", "call", "with", "without", "since", "before", "after", "never", "still", "until", "try", "ask", "state", "years","af", "become"]   \
        +en_stop

    preprocessed_docs = [clustering.preprocess_doc(document, en_stop) for document in docs]
    #print(" ".join(docs[0][:100]))
    #print(preprocessed_docs[0][:100])

    num_topics = cli_args().num_topics

    dictionary = corpora.Dictionary(preprocessed_docs)
    our_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'/workspaces/doc_analysis/output/{now}'
    os.makedirs(result_dir)

    with open(f'{result_dir}/token_id.json', mode='w') as f:
        json.dump(dictionary.token2id, f, indent=4)

    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus=our_corpus,
        num_topics=num_topics,
        id2word=dictionary,
        alpha=0.1,
        eta=0.1,
        minimum_probability=0.01,
        random_state=314
    )

    # visualize the result of LDA
    lda_display = pyLDAvis.gensim_models.prepare(ldamodel, our_corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, f'{result_dir}/{num_topics}-lda.html')
    #pyLDAvis.show(lda_display)

    # show keywords of each topic
    topics = ldamodel.print_topics(num_words=10)
    topic_dict = {}
    for (topic_id, distribution) in topics:
        topic_dict[topic_id] = distribution
    with open(f'{result_dir}/{num_topics}-topic.json', mode='w') as f:
        json.dump(topic_dict, f, indent=4)

    # show topic distribution of ten documents
    doc_topic = {}
    random.seed(314)
    id_list = random.sample(range(500), k=10)
    for doc_id in id_list:
        doc_topic[doc_id] = {
            'distribution':str(ldamodel.get_document_topics(our_corpus[doc_id])),
            'category':categories[doc_id],
            'text':" ".join(docs[doc_id])
        }
        #print(doc_topic[doc_id])
    with open(f'{result_dir}/{num_topics}-document.json', mode='w') as f:
        json.dump(doc_topic, f, indent=4)


if __name__ == "__main__":
    main()