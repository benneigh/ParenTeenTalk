import json
import re

ADVICE_PHRASES = [
    r'\byou should\b', r'\byou could\b', r'\btry\b', r'\bconsider\b',
    r'\bmy advice\b', r'\bi suggest\b', r'\bi recommend\b', r'\bif I were you\b',
    r'\bmy recommendation\b', r'\bthe best option\b', r'\bthe best way\b'
]

def extract_advice(post_content):

    if not post_content:
        return []
    
    sentences = re.split(r'(?<=[.!?]) +', post_content)
    advice_sentences = []

    for sentence in sentences:
        if any(re.search(phrase, sentence, re.IGNORECASE) for phrase in ADVICE_PHRASES):
            advice_sentences.append(sentence)
    
    return advice_sentences

def display_advice(input_file):
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    advice_count = 0
    for post in posts:
        content = post.get("content", "")
        advice = extract_advice(content)
        
        if advice:  
            advice_count += 1
            # print(f"Post Title: {post.get('title', 'No Title')}")
            # print("Advice found:")
            # for sentence in advice:
            #     print(f" - {sentence}")
            # print("-" * 40)
    
    total_posts = len(posts)
    advice_percentage = (advice_count / total_posts) * 100 if total_posts > 0 else 0
    print(f"\nTotal number of posts with advice: {advice_count}/{total_posts} = {advice_percentage:.2f}%")

input_file = 'r_parenting.json'  

display_advice(input_file)