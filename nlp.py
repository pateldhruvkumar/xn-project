# -------------------------
# 0) PACKAGES & SETUP
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter

# NLP libraries
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Sklearn for vectorization and topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

# Yellowbrick for visualizations
from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import warnings
warnings.filterwarnings('ignore')

# Set style and seed
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
np.random.seed(123)

print("✓ All packages loaded successfully\n")

# -------------------------
# 1) LOAD DATA
# -------------------------
# Update this path for your environment
# For Kaggle: file_path = "/kaggle/input/your-dataset/FY19_to_FY23_Cleaned.xlsx"
file_path = "D:/Projects/xn-project/dataset/FY19_to_FY23_Cleaned.xlsx"
raw_df = pd.read_excel(file_path)

# Create document dataframe
docs = pd.DataFrame({
    'doc_id': [f'D{i+1}' for i in range(len(raw_df))],
    'text': raw_df['Summary Notes']
})

# Filter out missing/empty text
docs = docs[docs['text'].notna() & (docs['text'].str.strip() != '')]
docs = docs.reset_index(drop=True)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(docs.head())
print(f"\nDataset shape: {docs.shape}")
print("\nDataframe info:")
print(docs.info())
print("=" * 70 + "\n")

# -------------------------
# 2) EDA (Exploratory Data Analysis)
# -------------------------
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

def count_sentences(text):
    """Count sentences in text"""
    return len(sent_tokenize(str(text)))

def count_words(text):
    """Count words in text"""
    return len(word_tokenize(str(text)))

# Add text statistics
docs['n_sent'] = docs['text'].apply(count_sentences)
docs['n_word'] = docs['text'].apply(count_words)
docs['n_char'] = docs['text'].apply(len)

print("\nText Statistics:")
print(docs[['n_sent', 'n_word', 'n_char']].describe())

# Top tokens (raw, lowercased)
all_words = []
for text in docs['text']:
    words = word_tokenize(text.lower())
    all_words.extend(words)

word_freq = Counter(all_words)
top_tokens = pd.DataFrame(word_freq.most_common(10), columns=['token', 'count'])
print("\nTop 10 tokens (raw, lowercased):")
print(top_tokens)

# Visualize top tokens with Yellowbrick
from yellowbrick.text import FreqDistVisualizer

# Prepare text for Yellowbrick
all_docs_text = docs['text'].tolist()

# Create vectorizer and fit
vectorizer = CountVectorizer(lowercase=True, stop_words=None)
docs_vectorized = vectorizer.fit_transform(all_docs_text)

# Use Yellowbrick FreqDistVisualizer
fig, ax = plt.subplots(figsize=(12, 6))
visualizer = FreqDistVisualizer(
    features=vectorizer.get_feature_names_out(),
    n=25,
    ax=ax
)
visualizer.fit(docs_vectorized)
visualizer.finalize()
plt.title("Top 25 Tokens (Raw, Lowercased) - Yellowbrick Visualization", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("=" * 70 + "\n")

# -------------------------
# 3) PREPROCESSING (tokenize → stopwords → lemmatize)
# -------------------------
print("=" * 70)
print("TEXT PREPROCESSING")
print("=" * 70)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize"""
    # Tokenize and lowercase
    tokens = word_tokenize(str(text).lower())
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# Apply preprocessing
print("Preprocessing documents...")
docs['tokens'] = docs['text'].apply(preprocess_text)
docs['processed_text'] = docs['tokens'].apply(lambda x: ' '.join(x))

# Create token dataframe for analysis
tokens_df = []
for idx, row in docs.iterrows():
    for token in row['tokens']:
        tokens_df.append({'doc_id': row['doc_id'], 'word': token})
tokens_df = pd.DataFrame(tokens_df)

print(f"Total tokens after preprocessing: {len(tokens_df)}")
print(f"Unique tokens: {tokens_df['word'].nunique()}")
print("\nSample preprocessed tokens:")
print(tokens_df.head(20))

# Visualize top tokens after preprocessing with Yellowbrick
vectorizer_clean = CountVectorizer()
docs_clean_vectorized = vectorizer_clean.fit_transform(docs['processed_text'])

fig, ax = plt.subplots(figsize=(12, 6))
visualizer = FreqDistVisualizer(
    features=vectorizer_clean.get_feature_names_out(),
    n=25,
    ax=ax,
    color='green'
)
visualizer.fit(docs_clean_vectorized)
visualizer.finalize()
plt.title("Top 25 Tokens (After Preprocessing) - Yellowbrick Visualization", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("=" * 70 + "\n")

# -------------------------
# 4) ENCODING (BoW & TF-IDF)
# -------------------------
print("=" * 70)
print("TEXT ENCODING: BAG OF WORDS & TF-IDF")
print("=" * 70)

# Bag of Words
bow = tokens_df.groupby(['doc_id', 'word']).size().reset_index(name='n')
bow = bow.sort_values('n', ascending=False)

print("\nBag of Words (top 20 term frequencies):")
print(bow.head(20))

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(docs['processed_text'])

# Create TF-IDF dataframe
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = []
for idx, doc_id in enumerate(docs['doc_id']):
    tfidf_scores = tfidf_matrix[idx].toarray()[0]
    for word_idx, score in enumerate(tfidf_scores):
        if score > 0:
            tfidf_df.append({
                'doc_id': doc_id,
                'word': feature_names[word_idx],
                'tf_idf': score
            })

tfidf_df = pd.DataFrame(tfidf_df).sort_values('tf_idf', ascending=False)

print("\nTop TF-IDF terms per document (first 20 rows):")
top_tfidf_per_doc = tfidf_df.groupby('doc_id').head(5)
print(top_tfidf_per_doc.head(20))

# Plot top TF-IDF terms overall
top_tfidf_words = tfidf_df.groupby('word')['tf_idf'].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_tfidf_words)), top_tfidf_words.values, color='steelblue')
plt.yticks(range(len(top_tfidf_words)), top_tfidf_words.index)
plt.xlabel('Total TF-IDF Score', fontsize=12)
plt.ylabel('Terms', fontsize=12)
plt.title('Top 15 Terms by Total TF-IDF', fontsize=14, pad=20)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("=" * 70 + "\n")

# -------------------------
# 5) PRE-TRAINED WORD EMBEDDINGS (GloVe 6B, 50d)
# -------------------------
print("=" * 70)
print("PRE-TRAINED WORD EMBEDDINGS (GloVe)")
print("=" * 70)

# For Kaggle, update path to: "/kaggle/input/glove-dataset/glove.6B.50d.txt"
emb_path = Path("data/glove.6B.50d.txt")

if not emb_path.exists():
    print("\n⚠️  Missing 'data/glove.6B.50d.txt'")
    print("Please download GloVe 6B from: https://nlp.stanford.edu/projects/glove/")
    print("Or add GloVe dataset in Kaggle via '+ Add Data'\n")
    embeddings = None
else:
    print("Loading GloVe embeddings (this may take a moment)...")
    
    # Load GloVe embeddings
    embeddings = {}
    with open(emb_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    
    print(f"✓ Embeddings loaded: {len(embeddings)} words × {len(next(iter(embeddings.values())))} dims\n")
    
    # Cosine similarity function
    from scipy.spatial.distance import cosine
    
    def cosine_similarity_top_n(embeddings, term, topn=5):
        """Find most similar words using cosine similarity"""
        term = term.lower()
        if term not in embeddings:
            print(f"Term '{term}' not found in embeddings")
            return pd.DataFrame({'term': [], 'similarity': []})
        
        term_vec = embeddings[term]
        similarities = {}
        
        for word, vec in embeddings.items():
            if word != term:
                sim = 1 - cosine(term_vec, vec)
                similarities[word] = sim
        
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:topn]
        return pd.DataFrame(top_similar, columns=['term', 'similarity'])
    
    print("Nearest neighbors (pre-trained GloVe 50d):\n")
    
    # Test with multiple relevant terms
    test_terms = ['email', 'meeting', 'review', 'project']
    for term in test_terms:
        if term in embeddings:
            print(f"\nTop 5 similar words to '{term}':")
            print(cosine_similarity_top_n(embeddings, term, 5))

print("=" * 70 + "\n")

# -------------------------
# 6) DOCUMENT CLUSTERING WITH YELLOWBRICK
# -------------------------
print("=" * 70)
print("DOCUMENT CLUSTERING VISUALIZATION")
print("=" * 70)

# Use TF-IDF matrix for clustering
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# K-Elbow Visualizer to find optimal k
print("\nFinding optimal number of clusters using Elbow method...")
fig, ax = plt.subplots(figsize=(10, 6))
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 12), ax=ax, timings=False)
visualizer.fit(tfidf_matrix.toarray())
visualizer.finalize()
plt.title("K-Elbow Visualizer for Optimal Clusters", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

optimal_k = visualizer.elbow_value_
if optimal_k is None:
    optimal_k = 5
print(f"Suggested optimal k: {optimal_k}")

# Silhouette Visualizer
print(f"\nVisualizing Silhouette scores for k={optimal_k}...")
fig, ax = plt.subplots(figsize=(10, 6))
model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
visualizer.fit(tfidf_matrix.toarray())
visualizer.finalize()
plt.title(f"Silhouette Plot for K-Means Clustering (k={optimal_k})", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Document clustering with t-SNE visualization
print("\nVisualizing document clusters using t-SNE...")
from yellowbrick.text import TSNEVisualizer

fig, ax = plt.subplots(figsize=(12, 8))
tsne = TSNEVisualizer(
    ax=ax,
    random_state=42,
    colormap='viridis'
)
tsne.fit(tfidf_matrix.toarray(), docs['doc_id'])
tsne.finalize()
plt.title("t-SNE Visualization of Document Similarity", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("=" * 70 + "\n")

# -------------------------
# 7) TOPIC MODELING (LDA, k = 3)
# -------------------------
print("=" * 70)
print("TOPIC MODELING WITH LDA")
print("=" * 70)

# Create document-term matrix for LDA
count_vectorizer = CountVectorizer(max_features=1000, min_df=2)
dtm = count_vectorizer.fit_transform(docs['processed_text'])

print(f"\nDocument-Term Matrix shape: {dtm.shape}")
print("Sample of DTM (first 2 docs, first 5 terms):")
print(dtm[:2, :5].toarray())

# Fit LDA model
n_topics = 4
print(f"\nFitting LDA model with {n_topics} topics...")

lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)
lda_model.fit(dtm)

# Get feature names
feature_names_lda = count_vectorizer.get_feature_names_out()

# Display top words per topic
n_top_words = 8
print(f"\nTop {n_top_words} words per topic (LDA, k={n_topics}):\n")

topic_words_data = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[-n_top_words:][::-1]
    top_words = [feature_names_lda[i] for i in top_indices]
    top_scores = topic[top_indices]
    
    print(f"Topic {topic_idx + 1}:")
    for word, score in zip(top_words, top_scores):
        print(f"  {word}: {score:.4f}")
        topic_words_data.append({
            'topic': topic_idx + 1,
            'term': word,
            'beta': score
        })
    print()

topic_words_df = pd.DataFrame(topic_words_data)

# Visualize topics
fig, axes = plt.subplots(1, n_topics, figsize=(18, 6))

for topic_idx in range(n_topics):
    topic_data = topic_words_df[topic_words_df['topic'] == topic_idx + 1]
    topic_data = topic_data.sort_values('beta', ascending=True)
    
    ax = axes[topic_idx] if n_topics > 1 else axes
    ax.barh(topic_data['term'], topic_data['beta'], color='steelblue')
    ax.set_title(f'Topic {topic_idx + 1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('β (word probability)', fontsize=10)
    ax.tick_params(axis='y', labelsize=9)

plt.suptitle('Top Words per Topic (LDA)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Document-topic distribution
doc_topic_dist = lda_model.transform(dtm)
docs['dominant_topic'] = doc_topic_dist.argmax(axis=1) + 1

print("\nDocument-Topic Distribution Summary:")
print(docs['dominant_topic'].value_counts().sort_index())

# Visualize document-topic distribution
plt.figure(figsize=(10, 6))
topic_counts = docs['dominant_topic'].value_counts().sort_index()
plt.bar(topic_counts.index, topic_counts.values, color='steelblue', alpha=0.7)
plt.xlabel('Topic', fontsize=12)
plt.ylabel('Number of Documents', fontsize=12)
plt.title('Distribution of Documents Across Topics', fontsize=14, pad=20)
plt.xticks(range(1, n_topics + 1))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("=" * 70)
print("\n✓ Analysis Complete!")
print("=" * 70)

# ============================================================================
# End of Script
# ============================================================================