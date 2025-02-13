import json
import re

QUESTION_WORDS = ["what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "should", "can", "could", "would"]

def contains_question(post_content):
    if not post_content:
        return False
    
    if "?" in post_content:
        return True

    pattern = r'\b(' + '|'.join(QUESTION_WORDS) + r')\b'
    return bool(re.search(pattern, post_content, re.IGNORECASE))

def count_questions(input_file):
    with open(input_file, 'r') as f:
        posts = json.load(f)
    
    question_count = sum(1 for post in posts if contains_question(post.get("content", "")))
    total_posts = len(posts)
    question_percentage = (question_count / total_posts) * 100 if total_posts > 0 else 0
    
    print(f"Total number of posts with questions: {question_count}/{total_posts} = {question_percentage:.2f}%")

input_file = 'r_parenting.json'  

count_questions(input_file)