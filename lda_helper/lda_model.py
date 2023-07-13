import spacy
import numpy as np
import random 
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model 
nlp = spacy.load('en_core_web_sm')

# Lemmatize
def lemmatize(text):
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    return text

# Preprocess
def preprocess(text):
    text = text.lower() 
    text = lemmatize(text)
    return text

# Remove stopwords and punctuation
def remove_stopwords_and_punctuation(doc):
    doc = [word for word in doc if word not in STOP_WORDS and word.isalpha()]
    return doc

# Generate bag of words
def make_bow(doc):
    doc = doc.split()
    doc = remove_stopwords_and_punctuation(doc)
    word_to_id = {word: i for i, word in enumerate(set(doc))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    return [word_to_id[word] for word in doc], id_to_word

# Run LDA with Gibbs sampling
def run_lda(doc, K=5, alpha=0.1, beta=0.1, n_samples=2000, burn_in=1000):
    # Number of documents (just 1 for now)
    M = 1

    # Vocabulary size
    V = len(set(doc)) 

    # Initialize model
    z = [random.randint(0, K-1) for _ in doc] # topic assignments
    theta = np.zeros((M, K)) + alpha # document-topic distributions
    phi = np.zeros((K, V)) + beta # topic-word distributions
    topic_counts = np.zeros((M, K)) + alpha
    word_counts = np.zeros((K, V)) + beta

    # Run sampling
    for i in range(n_samples):
        for j, w in enumerate(doc):
            # decrement the current count
            topic_counts[0, z[j]] -= 1
            word_counts[z[j], w] -= 1

            # calculate the conditional distribution
            p_z = (topic_counts[0] / topic_counts[0].sum()) * (word_counts[:, w] / word_counts.sum(axis=1))

            # Sample topic assignment z
            z[j] = np.random.choice(K, p=p_z/p_z.sum())

            # increment the new count
            topic_counts[0, z[j]] += 1
            word_counts[z[j], w] += 1

        if i >= burn_in and i % 100 == 0:
            # Record topic distributions
            theta += topic_counts

    # Normalize theta
    theta /= theta.sum(axis=1, keepdims=True)

    # Output topics and top words
    topic_words = []
    for k in range(K):
        idx = np.argsort(word_counts[k,:])[::-1]
        topic_words.append([id_to_word[i] for i in idx[:10]])

    return theta