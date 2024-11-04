import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if text is None:
        text = ""
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def prepare_corpus(posts):

    processed_posts = [preprocess_text(post.get("content", "")) for post in posts]

    dictionary = corpora.Dictionary(processed_posts)
    corpus = [dictionary.doc2bow(text) for text in processed_posts]
    
    return dictionary, corpus

def display_topics(lda_model, dictionary, num_topics=5, num_words=5):

    for i, topic in enumerate(lda_model.print_topics(num_topics=num_topics, num_words=num_words)):
        print(f"Topic #{i + 1}:")
        
        terms = topic[1].split(" + ")
        keywords = [term.split('*')[1].replace('"', '').strip() for term in terms]
        
        print(" + ".join(keywords))
        print()

def count_posts_per_topic(lda_model, corpus, num_topics=5):

    topic_counts = Counter()
    for doc_bow in corpus:

        topic_distribution = lda_model.get_document_topics(doc_bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        topic_counts[dominant_topic] += 1
    
    for topic_id in range(num_topics):
        print(f"Topic #{topic_id + 1} has {topic_counts[topic_id]} posts")

def analyze_topics(input_file, num_topics=5, num_words=5):
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    dictionary, corpus = prepare_corpus(posts)
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    
    display_topics(lda_model, dictionary, num_topics, num_words)
    
    count_posts_per_topic(lda_model, corpus, num_topics)

input_file = 'r_teenagers.json' 

analyze_topics(input_file, num_topics=5, num_words=5)