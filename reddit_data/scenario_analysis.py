import json
import spacy

nlp = spacy.load("en_core_web_sm")

def is_scenario(post_content):
    if not post_content:
        return False
    
    doc = nlp(post_content)
    
    pronoun_found = False
    action_verb_found = False
    
    for token in doc:

        if token.pos_ == "PRON" and token.text.lower() in {"i", "we", "my", "our"}:
            pronoun_found = True
       
        if token.pos_ == "VERB" and token.tag_ in {"VBD", "VBP", "VBZ", "VBG"}:
            action_verb_found = True
        

        if pronoun_found and action_verb_found:
            return True
    
    return False

def count_scenarios(input_file):

    with open(input_file, 'r') as f:
        posts = json.load(f)
    

    scenario_count = sum(1 for post in posts if is_scenario(post.get("content", "")))
    total_posts = len(posts)
    scenario_percentage = (scenario_count / total_posts) * 100 if total_posts > 0 else 0
    
    print(f"Total number of posts with scenarios: {scenario_count}/{total_posts} = {scenario_percentage:.2f}%")


input_file = 'r_parenting.json'  # Replace with your input file path

count_scenarios(input_file)