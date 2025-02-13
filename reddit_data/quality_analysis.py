import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

def classify_quality(post_content):
    if not post_content:
        return False
    
    sentiment_scores = sid.polarity_scores(post_content)
    return sentiment_scores['neg'] > 0.5

def count_rants(input_file):
    # Load the JSON data
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    total_posts = len(posts)
    rant_count = sum(1 for post in posts if classify_quality(post.get("content", "")))
    
    rant_percentage = (rant_count / total_posts) * 100 if total_posts > 0 else 0

    print(f"Total number of rants: {rant_count}/{total_posts} = {rant_percentage:.2f}%")

input_file = 'r_teenagers.json' 

count_rants(input_file)