# %%
from calendar import c
import re
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download("reuters")
# %%
from nltk.corpus import reuters
import numpy as np

# %%
reuters.fileids()[:10]

# %%
documents = reuters.fileids()
print(reuters.words(documents[0]))  # Words in a document

# %%


# custom implementation of LDA using collapsed gibbs sampling
def lda(corpus, num_topics, num_words, num_iterations, alpha=0.1, beta=0.1):
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab = tokenizer.get_vocab()

# Preprocess the corpus using huggingface tokenizer
corpus = [
    [tokenizer.convert_tokens_to_ids(w) for w in reuters.words(fileid)]
    for fileid in reuters.fileids()[:50]
]

num_words = len(vocab)
num_topics = 30
num_iterations = 200

assignments, doc_topic_counts, topic_word_counts = lda(
    corpus, num_topics, num_words, num_iterations, alpha=0.01, beta=0.01
)

print("Document-Topic Counts:\n", doc_topic_counts)
print("Topic-Word Counts:\n", topic_word_counts)

# %%

# Assess model performance in tables over the twenty top words for the topics that seem to make the most sense.

top_n = 20
inv_vocab = {v: k for k, v in vocab.items()}

for topic_idx in range(num_topics):
    # get top n words for each topic
    top_word_indices = topic_word_counts[topic_idx].argsort()[-top_n:][::-1]
    top_words = [inv_vocab[idx] for idx in top_word_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# %%

############### Build vocabulary from Reuters Corpus

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)

# create subset for testing (full corpus is 10,788 documents)
num_docs = 2000
subset_docs = reuters.fileids()[:num_docs]
raw_docs = [reuters.raw(f) for f in subset_docs]

tokenized_docs = []
for doc in raw_docs:
    tokens = [w.lower() for w in word_tokenize(doc)]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    # if len(tokens) > 0:
    tokenized_docs.append(tokens)

raw_docs[0], tokenized_docs[0]

from collections import Counter

word_freq = Counter()
for doc in tokenized_docs:
    word_freq.update(doc)

threshold = 4
vocab = {w for w, c in word_freq.items() if c >= threshold}

word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}

corpus = [[word2id[w] for w in doc if w in word2id] for doc in tokenized_docs]

num_docs = len(corpus)
total_tokens = sum(len(doc) for doc in corpus)

print("Number of documents:", num_docs)
print("Total tokens after preprocessing:", total_tokens)
print("Vocabulary size: ", len(vocab))

# %%
num_topics = 20
num_words = len(vocab)
num_iterations = 150

assignments, doc_topic_counts, topic_word_counts = lda(
    corpus, num_topics, num_words, num_iterations, alpha=0.1, beta=0.1
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

# total counts of each word in corpus
total_word_counts = topic_word_counts.sum(axis=0)  # shape: (num_words,)

# %%

# total counts of each word in corpus
total_word_counts = topic_word_counts.sum(axis=0)  # shape: (num_words,)

# top 20 words per topic by relative frequency
top_n = 20
for topic_idx in range(num_topics):
    # get top n words for each topic
    relative_freq = topic_word_counts[topic_idx] / total_word_counts
    top_word_indices = relative_freq.argsort()[-top_n:][::-1]
    top_words = [id2word[idx] for idx in top_word_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")


# %%
# Comparison to actual Reuters categories
category_topic_counts = np.zeros((len(reuters.categories()), num_topics))

for doc_idx, fileid in enumerate(subset_docs):
    categories = reuters.categories(fileid)
    for category in categories:
        category_idx = reuters.categories().index(category)
        # find argmax topic for the document
        # max_topic_idx = doc_topic_counts[doc_idx].argmax()
        # category_topic_counts[category_idx][max_topic_idx] += 1
        category_topic_counts[category_idx] += doc_topic_counts[doc_idx]


print("Category-Topic Counts:\n", category_topic_counts)

# show confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
sns.heatmap(
    category_topic_counts,
    xticklabels=[f"Topic {i}" for i in range(num_topics)],
    yticklabels=reuters.categories(),
    cmap="YlGnBu",
    annot=True,
    fmt=".0f",
)

print("Count per topic:", category_topic_counts.sum(axis=0))

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
