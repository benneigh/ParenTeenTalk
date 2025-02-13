from dotenv import load_dotenv
import os
import openai
import json
from better_profanity import profanity
import pandas as pd
import re
import time

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

profanity.load_censor_words()

def clean_text(text):
    return profanity.censor(text)

def ask_llm_for_topics(batch_text):
    prompt = (
        "Analyze the following text and summarize the main themes in the format:\n\n"
        "- Theme: [Theme Name]\n"
        "  - Description: Brief description\n"
        "  - Count: Number of posts under this theme\n\n"
        "Please ensure consistent formatting.\n\n"
        f"{batch_text}\n\n"
        "Provide a list of themes and their respective counts."
    )
    
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt,
                max_tokens=500
            )
            return response['choices'][0]['text']
        except openai.error.InvalidRequestError as e:
            print(f"Request failed due to content policy: {e}")
            if retry_count == max_retries - 1:
                print("Skipping batch due to repeated content policy issues.")
                return "Skipped due to content policy"
            retry_count += 1
            time.sleep(1) 
        except Exception as e:
            print(f"Unexpected error: {e}")
            break 
    
    return "Failed to retrieve topics"

def parse_topics_response(response_text):
    themes = []
    theme_name, description, count = None, None, 0  

    for line in response_text.split('\n'):
        if line.startswith("- Theme:"):
            if theme_name and description is not None:  
                themes.append({"Theme": theme_name, "Description": description, "Count": count})
            theme_name = line.split(":")[1].strip()
            description, count = None, 0  
        elif line.startswith("  - Description:"):
            description = line.split(":")[1].strip()
        elif line.startswith("  - Count:"):
            match = re.search(r'\d+', line)
            count = int(match.group()) if match else 0  
    
    if theme_name and description is not None:
        themes.append({"Theme": theme_name, "Description": description, "Count": count})
        
    return themes

def get_topic_model_from_posts(input_file, batch_size=50, output_file='topics_summary.xlsx'):
    with open(input_file, 'r') as f:
        posts = json.load(f)

    all_themes = []

    for i in range(0, len(posts), batch_size):
        batch_posts = posts[i:i + batch_size]
        
        batch_text = "\n\n".join([clean_text(post.get("content", ""))[:300] for post in batch_posts if "content" in post])

        print(f"Processing batch {i // batch_size + 1}...")
        topics_response = ask_llm_for_topics(batch_text)
        # print(f"Topics in batch {i // batch_size + 1}: {topics_response}\n")

        themes = parse_topics_response(topics_response)
        all_themes.extend(themes)

    df = pd.DataFrame(all_themes)
    df_grouped = df.groupby("Theme").agg({"Count": "sum", "Description": "first"}).reset_index()


    df_grouped.to_excel(output_file, index=False)

    print(f"Overall Topics Summary saved to {output_file}\n")
    
    return df_grouped


input_file = 'r_teenagers.json'
output_file = 'topics_summary.xlsx'
topics_df = get_topic_model_from_posts(input_file, batch_size=50, output_file=output_file)