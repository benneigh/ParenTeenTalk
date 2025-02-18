import pandas as pd
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langchain_openai import ChatOpenAI
from transformers import pipeline
from detoxify import Detoxify
from bert_score import score
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
    topics_df = pd.read_excel(topics_csv) 
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

# Finalized
def compute_semantic_similarity(dialogue_df):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    similarities = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        conversation_text = " ".join(group['utterance'].astype(str).tolist())
        
        topic_embedding = model.encode(topic_text, convert_to_tensor=True)
        conversation_embedding = model.encode(conversation_text, convert_to_tensor=True)
        
        similarity_score = round(util.pytorch_cos_sim(topic_embedding, conversation_embedding).item(), 4)
        similarities.append({'Variant_ID': variant_id, 'Semantic_Similarity': similarity_score})
    
    return pd.DataFrame(similarities)

# Finalized
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

# Finalized
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


# # Change to provide all of the numbers (sub categories) and we will calculate the average ourselves
# def compute_developmental_analysis(dialogue_df, attributes_df):
#     api_key = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv("OPENAI_API_BASE")
#     client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
    
#     developmental_scores = []
    
#     for variant_id, group in dialogue_df.groupby('Variant_ID'):
#         parent_text = " ".join(group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist())
#         child_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Child_Attributes']

#         # Check if it's empty or NaN
#         if child_attributes.empty or pd.isna(child_attributes.iloc[0]):
#             child_attributes = "{}"
#         else:
#             child_attributes = child_attributes.iloc[0]  # Extract the value

#         child_age = json.loads(child_attributes).get('age', {}).get('value', None)
#         topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        
#         prompt = f"""
#         You are tasked with evaluating the developmental appropriateness of a conversation between a parent and a child.
#         Your response should only be a number.
#         The conversation is about the topic: {topic_text}.
#         """
        
#         if child_age:
#             prompt += f"\nThe child's age is {child_age} years old."
        
#         prompt += "\n\nEvaluate the conversation based on the following criteria and provide a score from 1 to 5 for each:\n"
#         prompt += "1. Appropriateness for child's age\n2. Response length\n3. Depth of detail on a single topic (more for older children)\n"
#         prompt += "4. Breadth (more topics = lower score)\n5. Level of concreteness vs. abstraction (younger = more concrete, older = more abstract)\n\n"
#         prompt += f"Here is the parent's part of the conversation:\n{parent_text}\n\nOnly provide the average of the scores without reasoning."
        
#         response = client.invoke(prompt).content.strip()
        
#         developmental_scores.append({
#             'Variant_ID': variant_id,
#             'Developmental_Analysis': response
#         })
    
#     return pd.DataFrame(developmental_scores)


# # TODO Change to latest Guidelines (top table) and also breakdown of each score and we will calc average
# def compute_communication_skills_analysis(dialogue_df, attributes_df):
#     api_key = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv("OPENAI_API_BASE")
#     client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
    
#     communication_scores = []
    
#     for variant_id, group in dialogue_df.groupby('Variant_ID'):
#         parent_text = " ".join(group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist())
#         child_attributes = attributes_df.loc[attributes_df['ID'] == variant_id, 'Child_Attributes']

#         # Check if child attributes exist
#         if child_attributes.empty or pd.isna(child_attributes.iloc[0]):
#             child_attributes = "{}"
#         else:
#             child_attributes = str(child_attributes.iloc[0])  # Ensure it's a string

#         child_age = json.loads(child_attributes).get('age', {}).get('value', None)
#         topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        
#         prompt = f"""
#         You are evaluating a parent's communication skills based on expert guidelines.
#         Your response should only be a number.
#         The conversation is about the topic: {topic_text}.
#         """

#         if child_age:
#             prompt += f"\nThe child's age is {child_age} years old."

#         prompt += """
#         Evaluate the conversation based on the following guidelines, and provide a score from 1 to 5 for each:
        
        
#         1. Ask open-ended questions.
#         2. Don’t jump to conclusions; ask what they already know before responding.
#         3. Keep answers short and simple, explaining new words if needed.
#         4. Keep the conversation open after answering a question.
#         5. Check for understanding by asking follow-up questions.

#         **Parent’s Conversation:**
#         {parent_text}
        
#         Only provide the average of the scores without any explanation.
#         """

#         response = client.invoke(prompt).content.strip()
        
#         communication_scores.append({
#             'Variant_ID': variant_id,
#             'Communication_Skills_Score': response
#         })
    
#     return pd.DataFrame(communication_scores)


# Finalized
def compute_parenting_analysis(dialogue_df, attributes_df):
    """Computes both developmental appropriateness and communication skills in a single function."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
    
    analysis_results = []
    
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
        You are evaluating a parent's conversation with their child for both developmental appropriateness and communication skills.
        
        The conversation is about the topic: {topic_text}.
        The child's age is {child_age} years old.
        
        **Developmental Appropriateness Criteria (Rate 1-5)**
        1. Appropriateness for child's age
        2. Response length
        3. Depth of detail on a single topic (more for older children)
        4. Breadth (more topics = lower score)
        5. Level of concreteness vs. abstraction (younger = more concrete, older = more abstract)
        
        **Communication Skills Criteria (Rate 1-5)**
        6. Asking open-ended questions
        7. Avoiding assumptions, first checking child's knowledge
        8. Keeping responses short and simple
        9. Keeping the conversation open after answering a question
        10. Checking for understanding with follow-up questions

        **Parent’s Conversation:**
        {parent_text}
        
        Provide a **list of 10 numerical scores (1-5)** in the exact order above, **separated by commas** (e.g., "5,4,3,4,5,5,3,4,5,4").
        """
        
        response = client.invoke(prompt).content.strip()
        scores = [int(x) for x in response.split(",")] if "," in response else [None] * 10

        # Calculate average scores
        avg_developmental_score = round(sum(scores[:5]) / 5, 4) if all(scores[:5]) else None
        avg_communication_score = round(sum(scores[5:]) / 5, 4) if all(scores[5:]) else None

        analysis_results.append({
            'Variant_ID': variant_id,
            'Dev_Age_Appropriateness': scores[0],
            'Dev_Response_Length': scores[1],
            'Dev_Depth_of_Topic': scores[2],
            'Dev_Breadth_of_Topics': scores[3],
            'Dev_Concreteness_vs_Abstraction': scores[4],
            'Comm_Open_Ended_Questions': scores[5],
            'Comm_Avoiding_Assumptions': scores[6],
            'Comm_Short_and_Simple': scores[7],
            'Comm_Keeping_Conversation_Open': scores[8],
            'Comm_Checking_Understanding': scores[9],
            'Avg_Developmental_Score': avg_developmental_score,
            'Avg_Communication_Score': avg_communication_score
        })

    return pd.DataFrame(analysis_results)


# Finalized
def compute_descriptive_statistics(dialogue_df, attributes_df):
    descriptive_results = []

    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        total_utterances = len(group)
        parent_utterances = group[group['speaker'] == 'Parent']
        child_utterances = group[group['speaker'] == 'Child']

        # Compute average utterance length (in words)
        parent_avg_length = (
            round(parent_utterances['utterance'].apply(lambda x: len(str(x).split())).mean(), 4)
            if not parent_utterances.empty else 0
        )
        child_avg_length = (
            round(child_utterances['utterance'].apply(lambda x: len(str(x).split())).mean(), 4)
            if not child_utterances.empty else 0
        )

        descriptive_results.append({
            'Variant_ID': variant_id,
            'Total_Utterances': total_utterances,
            'Parent_Avg_Utterance_Length': parent_avg_length,
            'Child_Avg_Utterance_Length': child_avg_length
        })

    return pd.DataFrame(descriptive_results)


nli_pipeline = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-large")

def check_entailment(premise, hypothesis):
    result = nli_pipeline(f"{premise} [SEP] {hypothesis}")
    return result[0]['label'], result[0]['score']

# TODO: Keep it as a percentage and also go turn by turn - Parent+Child and then Child+Parent
def compute_turn_level_entailment(dialogue_df):
    entailment_results = []
    
    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        turns = group[['speaker', 'utterance']].values.tolist()
        entailment_count = 0
        total_pairs = 0
        
        for i in range(1, len(turns)):
            premise = turns[i - 1][1]  # Previous utterance
            hypothesis = turns[i][1]  # Current utterance
            
            label, score = check_entailment(premise, hypothesis)
            
            if label == "ENTAILMENT":
                entailment_count += 1
            total_pairs += 1
            
        entailment_percentage = (entailment_count / total_pairs * 100) if total_pairs > 0 else 0
        
        entailment_results.append({
            'Variant_ID': variant_id,
            'Entailment_Percentage': entailment_percentage
        })
    
    return pd.DataFrame(entailment_results)

# Finalized
def compute_toxicity_score(dialogue_df):
    """Computes toxicity scores for Parent and Child utterances separately."""
    toxicity_results = []

    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        parent_utterances = group[group['speaker'] == 'Parent']['utterance'].astype(str).tolist()
        child_utterances = group[group['speaker'] == 'Child']['utterance'].astype(str).tolist()
        
        # Compute toxicity for Parent
        parent_toxicity = Detoxify('original').predict(parent_utterances) if parent_utterances else None
        child_toxicity = Detoxify('original').predict(child_utterances) if child_utterances else None

        # Compute average toxicity scores
        parent_avg_toxicity = round(np.mean(parent_toxicity['toxicity']), 4) if parent_toxicity else 0
        child_avg_toxicity = round(np.mean(child_toxicity['toxicity']), 4) if child_toxicity else 0

        toxicity_results.append({
            'Variant_ID': variant_id,
            'Parent_Avg_Toxicity': parent_avg_toxicity,
            'Child_Avg_Toxicity': child_avg_toxicity
        })

    return pd.DataFrame(toxicity_results)



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
    llm_df = compute_parenting_analysis(dialogue_df, attributes_df)
    descriptive_df = compute_descriptive_statistics(dialogue_df, attributes_df)
    entailment_df = compute_turn_level_entailment(dialogue_df)
    toxicity_df = compute_toxicity_score(dialogue_df)

    # test_premise = "I love programming in Python."
    # test_hypothesis = "Python is my favorite language to code in."

    # result = check_entailment(test_premise, test_hypothesis)
    # print(result) 

    
    similarity_df.to_csv("semantic_similarity_scores.csv", index=False)
    stopping_df.to_csv("stopping_criteria_scores.csv", index=False)
    readability_df.to_csv("readability_scores.csv", index=False)
    # developmental_df.to_csv("developmental_analysis_scores.csv", index=False)
    # communication_df.to_csv("communication_skills_scores.csv", index=False)
    llm_df.to_csv("parenting_analysis_scores.csv", index=False)
    descriptive_df.to_csv("descriptive_statistics.csv", index=False)
    entailment_df.to_csv("turn_level_entailment.csv", index=False)
    toxicity_df.to_csv("toxicity_scores.csv", index=False)
    print("Evaluation completed. Results saved to CSV files.")

if __name__ == "__main__":
    main()
