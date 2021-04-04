from rake_nltk import Rake, Metric
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

n_gram_range = (1, 3)
stop_words = "english"

doc = """
HMB443H1: Global Hidden Hunger will be offered this summer!

In this course, you will discuss global food insecurity and come up with tangible ways to address equity in health and access to food, poverty and metabolic diseases that relate to deficiencies and the 'hidden hunger' that leads to deterioration of health across populations. Hidden Hunger is preventable and this course begins to practically address these issues and is really for anyone who is interested in the concepts of health access, inequity and food security.
This course also features a virtual service learning component with a partner in British Columbia.

Prerequisites: 12 FCE complete, At least 0.5 HMB 300-Level Courses.

If you are interested in taking this course, please click on the link below. 
 
"""

# Maximal Marginal Relevance (MMR): MMR tries to minimize redundancy and maximize the diversity of results in text summarization tasks.

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):

        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names()

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n=15, diversity=0.5)

=======
from rake_nltk import Rake, Metric
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

n_gram_range = (1, 3)
stop_words = "english"

doc = """
HMB443H1: Global Hidden Hunger will be offered this summer!

In this course, you will discuss global food insecurity and come up with tangible ways to address equity in health and access to food, poverty and metabolic diseases that relate to deficiencies and the 'hidden hunger' that leads to deterioration of health across populations. Hidden Hunger is preventable and this course begins to practically address these issues and is really for anyone who is interested in the concepts of health access, inequity and food security.
This course also features a virtual service learning component with a partner in British Columbia.

Prerequisites: 12 FCE complete, At least 0.5 HMB 300-Level Courses.

If you are interested in taking this course, please click on the link below. 
 
"""

# Maximal Marginal Relevance (MMR): MMR tries to minimize redundancy and maximize the diversity of results in text summarization tasks.

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):

        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names()

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n=15, diversity=0.5)

