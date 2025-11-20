# %%
import nltk
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
    for it in range(num_iterations):
        for i in range(num_docs):
            for j, word in enumerate(corpus[i]):
                current_topic = assignments[i][j]

                # Decrement counts for current assignment
                doc_topic_counts[i][current_topic] -= 1
                topic_word_counts[current_topic][word] -= 1
                topic_counts[current_topic] -= 1

                # Compute probabilities for each topic
                topic_probs = np.zeros(num_topics)
                for k in range(num_topics):
                    topic_probs[k] = (
                        (doc_topic_counts[i][k] + alpha)
                        * (topic_word_counts[k][word] + beta)
                        / (topic_counts[k] + num_words * beta)
                    )

                # Normalize probabilities
                topic_probs /= np.sum(topic_probs)

                # Sample new topic
                new_topic = np.random.choice(num_topics, p=topic_probs)

                # Increment counts with new topic
                assignments[i][j] = new_topic
                doc_topic_counts[i][new_topic] += 1
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
num_topics = 10
num_iterations = 100

assignments, doc_topic_counts, topic_word_counts = lda(
    corpus, num_topics, num_words, num_iterations
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
