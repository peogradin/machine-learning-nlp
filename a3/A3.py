# %%
from calendar import c
import re
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.corpus import reuters
import numpy as np

nltk.download("reuters")


# %%
documents = reuters.fileids()
print(reuters.words(documents[0]))  # Words in a document

# %%
# custom implementation of LDA using collapsed gibbs sampling
def lda(corpus, num_topics, num_words, num_iterations, alpha=0.1, beta=0.01):
    # count number of documents and unique words
    num_docs = len(corpus)

    # Initialize topic assignments randomly
    assignments = np.random.randint(
        num_topics, size=(num_docs, max(len(doc) for doc in corpus))
    )

    # Count matrices
    doc_topic_counts = np.zeros((num_docs, num_topics))  # Document-Topic counts
    topic_word_counts = np.zeros((num_topics, num_words))  # Topic-Word counts
    topic_counts = np.zeros(num_topics)  # Total counts per topic

    # Initialize counts based on initial assignments
    for i in range(num_docs):
        for j, word in enumerate(corpus[i]):
            topic = assignments[i][j]
            doc_topic_counts[i][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1

    # Collapsed Gibbs sampling
    for it in tqdm(range(num_iterations), desc="LDA Gibbs Sampling"):
        for d in range(num_docs):
            for j, word in enumerate(corpus[d]):
                current_topic = assignments[d][j]

                # Decrement counts for current assignment
                doc_topic_counts[d][current_topic] -= 1
                topic_word_counts[current_topic][word] -= 1
                topic_counts[current_topic] -= 1

                # Compute probabilities for each topic (vectorized)
                topic_probs = (
                    (doc_topic_counts[d] + alpha)
                    * (topic_word_counts[:, word] + beta)
                    / (topic_counts + beta * num_words)
                )

                # Normalize probabilities
                topic_probs /= np.sum(topic_probs)

                # Sample new topic
                new_topic = np.random.choice(num_topics, p=topic_probs)

                # Increment counts with new topic
                assignments[d][j] = new_topic
                doc_topic_counts[d][new_topic] += 1
                topic_word_counts[new_topic][word] += 1
                topic_counts[new_topic] += 1  #

    return assignments, doc_topic_counts, topic_word_counts

# %%
############### Build vocabulary from Reuters Corpus

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)

# create subset for testing (full corpus is 10,788 documents)
num_docs = 4000
subset_docs = reuters.fileids()[:num_docs]
raw_docs = [reuters.raw(f) for f in subset_docs]

tokenized_docs = []
for doc in raw_docs:
    tokens = [w.lower() for w in word_tokenize(doc)]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokenized_docs.append(tokens)

word_freq = Counter()
for doc in tokenized_docs:
    word_freq.update(doc)

threshold = 4
vocab = {w for w, c in word_freq.items() if c >= threshold}

word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}

corpus = [[word2id[w] for w in doc if w in word2id] for doc in tokenized_docs]
corpus = [doc for doc in corpus if len(doc) > 0]

num_docs = len(corpus)
total_tokens = sum(len(doc) for doc in corpus)

print("Number of documents:", num_docs)
print("Total tokens after preprocessing:", total_tokens)
print("Vocabulary size: ", len(vocab))

num_empty = sum(1 for doc in corpus if len(doc) == 0)
print("Empty documents:", num_empty)

# %%
num_topics = 25
num_words = len(vocab)
num_iterations = 150

assignments, doc_topic_counts, topic_word_counts = lda(
    corpus, num_topics, num_words, num_iterations, alpha=0.01, beta=0.01
)

print("Document-Topic Counts:\n", doc_topic_counts)
print("Topic-Word Counts:\n", topic_word_counts)

# %%
########## MODEL PERFORMANCE ########
# top 20 words per topic
top_n = 20

for topic_idx in range(num_topics):
    # get top n words for each topic
    top_word_indices = topic_word_counts[topic_idx].argsort()[-top_n:][::-1]
    top_words = [id2word[idx] for idx in top_word_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# %%

# total counts of each word in corpus
total_word_counts = topic_word_counts.sum(axis=0)  # shape: (num_words,)

# %%
# UMass coherence for top 20 words
import math

# get set of word for each doc
# D_w lists number of docs containing w_l
doc_wordsets = [set(doc) for doc in corpus]
D_w = np.zeros(num_words)

# loop over all words, sum if that word appears in a doc: D_w[w] += 1
for w in range(num_words):
    D_w[w] = sum(1 for s in doc_wordsets if w in s)

# function to calc umass score for a topic
def umass_coherence_score(top_word_indices, D_w, doc_wordsets):
    umass_score = 0.0
    M = len(top_word_indices)

    for m in range(1, M):
        w_m = top_word_indices[m]
        for l in range(m):
            w_l = top_word_indices[l]
            D_wm_wl = sum(1 for s in doc_wordsets if w_m in s and w_l in s)
            umass_score += math.log((D_wm_wl + 1) / (D_w[w_l] + 1e-12))
    return umass_score


# calc umass scores for all topics
umass_scores = []
for topic_idx in range(num_topics):
    top_word_indices = topic_word_counts[topic_idx].argsort()[-top_n:][::-1]
    score = umass_coherence_score(top_word_indices, D_w, doc_wordsets)
    umass_scores.append(score)
    print(f"Topic {topic_idx} UMass coherence score: {score:.5f}")

print("Average coherence:", np.mean(umass_scores))
# %%
