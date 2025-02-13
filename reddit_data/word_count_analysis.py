import json
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

def count_words(input_file):
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    word_counts = Counter()
    
    for post in posts:
        content = post.get("content", "")
        tokens = preprocess_text(content)
        word_counts.update(tokens)
    
    for word, count in word_counts.most_common(100): 
        print(f"{word}: {count}")

input_file = 'r_parenting.json' 

count_words(input_file)