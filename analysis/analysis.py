import pandas as pd
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

""" Documentation for the logic of the model for semantic similarity:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com 
"""

def load_data(dialogue_csv, attributes_csv, topics_csv):
    dialogue_df = pd.read_csv(dialogue_csv)
    attributes_df = pd.read_csv(attributes_csv)
    topics_df = pd.read_excel(topics_csv)  # Assuming topics are in an Excel file
    return dialogue_df, attributes_df, topics_df

def extract_variant_id(dialogue_df):
    """Extracts topic, iteration, and variant ID while ignoring utterance IDs."""
    dialogue_df['Variant_ID'] = dialogue_df['ID'].apply(lambda x: "-".join(x.split('-')[:3]))  # T#-I#-V#
    dialogue_df['Topic_Number'] = dialogue_df['ID'].apply(lambda x: int(x.split('-')[0][1:]))  # Extract T#
    return dialogue_df

def map_topics(dialogue_df, topics_df):
    """Maps Topic_Number to actual Topic_Text from topics dataset."""
    topic_map = dict(zip(topics_df['Topic_Number'], topics_df['Topic_Text']))
    dialogue_df['Topic_Text'] = dialogue_df['Topic_Number'].map(topic_map)
    return dialogue_df

def compute_semantic_similarity(dialogue_df):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    similarities = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        conversation_text = " ".join(group['utterance'].astype(str).tolist())
        
        topic_embedding = model.encode(topic_text, convert_to_tensor=True)
        conversation_embedding = model.encode(conversation_text, convert_to_tensor=True)
        
        similarity_score = util.pytorch_cos_sim(topic_embedding, conversation_embedding).item()
        similarities.append({'Variant_ID': variant_id, 'Semantic_Similarity': similarity_score})
    
    return pd.DataFrame(similarities)

def compute_stopping_criteria(dialogue_df):
    stopping_results = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        child_disengagement = any(group['stopper_decision'] == 'stop: child disengagement')
        goal_met = any(group['stopper_decision'] == 'stop: goal met')
        mutual_agreement = any(group['stopper_decision'] == 'stop: mutual agreement')
        turn_limit_reached = any(group['stopper_decision'] == 'stop: limit reached')
        
        stopping_results.append({
            'Variant_ID': variant_id,
            'Child_Disengagement': child_disengagement,
            'Goal_Met': goal_met,
            'Mutual_Agreement': mutual_agreement,
            'Turn_Limit_Reached': turn_limit_reached
        })
    
    return pd.DataFrame(stopping_results)

def compute_readability(dialogue_df):
    readability_scores = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        parent_text = " ".join(group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist())
        child_text = " ".join(group[group['speaker'] == 'Child']['utterance'].astype(str).tolist())
        
        parent_reading_ease = flesch_reading_ease(parent_text) if parent_text else None
        parent_grade_level = flesch_kincaid_grade(parent_text) if parent_text else None
        
        child_reading_ease = flesch_reading_ease(child_text) if child_text else None
        child_grade_level = flesch_kincaid_grade(child_text) if child_text else None
        
        readability_scores.append({
            'Variant_ID': variant_id,
            'Parent_Flesch_Reading_Ease': parent_reading_ease,
            'Parent_Flesch_Kincaid_Grade': parent_grade_level,
            'Child_Flesch_Reading_Ease': child_reading_ease,
            'Child_Flesch_Kincaid_Grade': child_grade_level
        })
    
    return pd.DataFrame(readability_scores)

def compute_developmental_analysis(dialogue_df, attributes_df):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
    
    developmental_scores = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        parent_text = " ".join(group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist())
        child_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Child_Attributes']

        # Check if it's empty or NaN
        if child_attributes.empty or pd.isna(child_attributes.iloc[0]):
            child_attributes = "{}"
        else:
            child_attributes = child_attributes.iloc[0]  # Extract the value

        child_age = json.loads(child_attributes).get('age', {}).get('value', None)
        topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        
        prompt = f"""
        You are tasked with evaluating the developmental appropriateness of a conversation between a parent and a child.
        Your response should only be a number.
        The conversation is about the topic: {topic_text}.
        """
        
        if child_age:
            prompt += f"\nThe child's age is {child_age} years old."
        
        prompt += "\n\nEvaluate the conversation based on the following criteria and provide a score from 1 to 5 for each:\n"
        prompt += "1. Appropriateness for child's age\n2. Response length\n3. Depth of detail on a single topic (more for older children)\n"
        prompt += "4. Breadth (more topics = lower score)\n5. Level of concreteness vs. abstraction (younger = more concrete, older = more abstract)\n\n"
        prompt += f"Here is the parent's part of the conversation:\n{parent_text}\n\nOnly provide the average of the scores without reasoning."
        
        response = client.invoke(prompt).content.strip()
        
        developmental_scores.append({
            'Variant_ID': variant_id,
            'Developmental_Analysis': response
        })
    
    return pd.DataFrame(developmental_scores)

def compute_communication_skills_analysis(dialogue_df, attributes_df):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
    
    communication_scores = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        parent_text = " ".join(group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist())
        child_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Child_Attributes']

        # Check if child attributes exist
        if child_attributes.empty or pd.isna(child_attributes.iloc[0]):
            child_attributes = "{}"
        else:
            child_attributes = str(child_attributes.iloc[0])  # Ensure it's a string

        child_age = json.loads(child_attributes).get('age', {}).get('value', None)
        topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        
        prompt = f"""
        You are evaluating a parent's communication skills based on expert guidelines.
        Your response should only be a number.
        The conversation is about the topic: {topic_text}.
        """

        if child_age:
            prompt += f"\nThe child's age is {child_age} years old."

        prompt += """
        Evaluate the conversation based on the following guidelines, and provide a score from 1 to 5 for each:
        
        **CDC Guidelines**
        1. Be honest and direct when talking about sensitive subjects.
        2. Help the child make healthy choices while allowing them to make their own decisions.
        3. Respect the child’s opinions and make sure they feel heard.
        4. Be clear about goals and expectations but allow input in decision-making.
        
        **Planned Parenthood Guidelines**
        5. Ask open-ended questions.
        6. Don’t jump to conclusions; ask what they already know before responding.
        7. Keep answers short and simple, explaining new words if needed.
        8. Keep the conversation open after answering a question.
        9. Check for understanding by asking follow-up questions.
        10. If unsure about an answer, offer to look it up together.

        **Parent’s Conversation:**
        {parent_text}
        
        Only provide the average of the scores without any explanation.
        """

        response = client.invoke(prompt).content.strip()
        
        communication_scores.append({
            'Variant_ID': variant_id,
            'Communication_Skills_Score': response
        })
    
    return pd.DataFrame(communication_scores)

def compute_descriptive_statistics(dialogue_df, attributes_df):
    descriptive_results = []

    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        total_utterances = len(group)
        parent_utterances = group[group['speaker'] == 'Parent']
        child_utterances = group[group['speaker'] == 'Child']

        # Compute utterance counts
        parent_utterance_count = len(parent_utterances)
        child_utterance_count = len(child_utterances)

        # Compute average utterance length (in words)
        parent_avg_length = (
            parent_utterances['utterance'].apply(lambda x: len(str(x).split())).mean()
            if not parent_utterances.empty else 0
        )
        child_avg_length = (
            child_utterances['utterance'].apply(lambda x: len(str(x).split())).mean()
            if not child_utterances.empty else 0
        )

        # Extract Parent and Child Attributes
        parent_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Parent_Attributes']
        child_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Child_Attributes']

        # Check if attributes exist
        parent_attr_dict = json.loads(parent_attributes.iloc[0]) if not parent_attributes.empty and pd.notna(parent_attributes.iloc[0]) else {}
        child_attr_dict = json.loads(child_attributes.iloc[0]) if not child_attributes.empty and pd.notna(child_attributes.iloc[0]) else {}

        # Extract specific attributes (if available)
        parent_confidence = parent_attr_dict.get('confidence_level', {}).get('value', 'Unknown')
        parent_comfort = parent_attr_dict.get('comfort_level', {}).get('value', 'Unknown')
        parent_open_dialogue = parent_attr_dict.get('open_dialogue', {}).get('value', 'Unknown')

        child_age = child_attr_dict.get('age', {}).get('value', 'Unknown')
        child_closeness = child_attr_dict.get('parent_child_closeness_level', {}).get('value', 'Unknown')
        child_gender = child_attr_dict.get('gender', {}).get('value', 'Unknown')

        descriptive_results.append({
            'Variant_ID': variant_id,
            'Total_Utterances': total_utterances,
            'Parent_Utterance_Count': parent_utterance_count,
            'Child_Utterance_Count': child_utterance_count,
            'Parent_Avg_Utterance_Length': parent_avg_length,
            'Child_Avg_Utterance_Length': child_avg_length,
            'Parent_Confidence': parent_confidence,
            'Parent_Comfort': parent_comfort,
            'Parent_Open_Dialogue': parent_open_dialogue,
            'Child_Age': child_age,
            'Child_Closeness': child_closeness,
            'Child_Gender': child_gender
        })

    return pd.DataFrame(descriptive_results)



def main():
    dialogue_csv = "../rag_agent/dialogue_example.csv"
    attributes_csv = "../rag_agent/attributes_example.csv"
    topics_csv = "topics.xlsx"
    
    dialogue_df, attributes_df, topics_df = load_data(dialogue_csv, attributes_csv, topics_csv)
    dialogue_df = extract_variant_id(dialogue_df)
    dialogue_df = map_topics(dialogue_df, topics_df)
    similarity_df = compute_semantic_similarity(dialogue_df)
    stopping_df = compute_stopping_criteria(dialogue_df)
    readability_df = compute_readability(dialogue_df)
    # developmental_df = compute_developmental_analysis(dialogue_df, attributes_df)
    # communication_df = compute_communication_skills_analysis(dialogue_df, attributes_df)
    descriptive_df = compute_descriptive_statistics(dialogue_df, attributes_df)
    
    similarity_df.to_csv("semantic_similarity_scores.csv", index=False)
    stopping_df.to_csv("stopping_criteria_scores.csv", index=False)
    readability_df.to_csv("readability_scores.csv", index=False)
    # developmental_df.to_csv("developmental_analysis_scores.csv", index=False)
    # communication_df.to_csv("communication_skills_scores.csv", index=False)
    descriptive_df.to_csv("descriptive_statistics.csv", index=False)
    print("Evaluation completed. Results saved to CSV files.")

if __name__ == "__main__":
    main()
