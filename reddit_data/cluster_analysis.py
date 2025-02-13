import json
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  

def embed_posts(posts):

    contents = [post.get("content", "") if post.get("content") is not None else "" for post in posts]
    embeddings = model.encode(contents, show_progress_bar=True)
    return embeddings, contents

def cluster_posts(embeddings, num_clusters=5):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def extract_keywords(contents, labels, num_clusters=5, num_keywords=5):
    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(contents)
    feature_names = vectorizer.get_feature_names_out()
    
    keywords = {}
    for cluster_id in range(num_clusters):
        cluster_contents = [contents[i] for i in range(len(contents)) if labels[i] == cluster_id]
        cluster_matrix = vectorizer.transform(cluster_contents)
        word_counts = np.asarray(cluster_matrix.sum(axis=0)).flatten()
        
        top_indices = word_counts.argsort()[-num_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        keywords[cluster_id] = top_keywords
    
    return keywords

def analyze_clusters(input_file, num_clusters=5):
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    embeddings, contents = embed_posts(posts)
    
    labels, kmeans = cluster_posts(embeddings, num_clusters)
    
    keywords = extract_keywords(contents, labels, num_clusters)
    for cluster_id, words in keywords.items():
        print(f"\nCluster {cluster_id} Topic Keywords:")
        print(", ".join(words))

input_file = 'r_teenagers.json' 

analyze_clusters(input_file, num_clusters=5)