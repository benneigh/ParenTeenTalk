from dotenv import load_dotenv
import os
import openai
import json
from better_profanity import profanity

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

profanity.load_censor_words()

def clean_text(text):
    return profanity.censor(text)

def ask_llm_for_topics(batch_text):

    prompt = (
        "This data is being used for a research study on parenting discussions. "
        "Please analyze these posts and list the main themes or life challenges covered. "
        "Avoid including any inappropriate language in your response.\n\n"
        f"{batch_text}\n\n"
        "Provide a summary of each main theme and estimate the number of distinct topics."
    )
    
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=500
    )
    
    return response['choices'][0]['text']

def get_topic_model_from_posts(input_file, batch_size=50):
    with open(input_file, 'r') as f:
        posts = json.load(f)

    topics = []

    for i in range(0, len(posts), batch_size):
        batch_posts = posts[i:i + batch_size]
        
        batch_text = "\n\n".join([clean_text(post.get("content", "")[:300]) for post in batch_posts])

        print(f"Processing batch {i // batch_size + 1}...")
        topics_response = ask_llm_for_topics(batch_text)
        print(f"Topics in batch {i // batch_size + 1}: {topics_response}\n")

        topics.append(topics_response)

    overall_summary = "\n\n".join(topics)
    print("Overall Topics Summary:\n")
    print(overall_summary)

input_file = 'r_teenagers.json'

get_topic_model_from_posts(input_file, batch_size=50)