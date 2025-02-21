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
import time
import logging
from functools import partial

from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

""" Documentation for the logic of the model for semantic similarity:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com 
"""

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("progress_log.txt"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to console
    ]
)
logger = logging.getLogger(__name__)

def load_data(dialogue_csv, attributes_csv, topics_csv, rag_csv):
    logger.info("Loading datasets...")
    dialogue_df = pd.read_csv(dialogue_csv)
    attributes_df = pd.read_csv(attributes_csv)
    topics_df = pd.read_excel(topics_csv) 
    rag_df = pd.read_excel(rag_csv)
    logger.info("Datasets loaded successfully.")
    return dialogue_df, attributes_df, topics_df, rag_df

def extract_variant_id(dialogue_df):
    """Extracts topic, iteration, and variant ID while ignoring utterance IDs."""
    logger.info("Extracting Variant IDs...")
    dialogue_df['Variant_ID'] = dialogue_df['ID'].apply(lambda x: "-".join(x.split('-')[:3]))  # T#-I#-V#
    dialogue_df['Topic_Number'] = dialogue_df['ID'].apply(lambda x: int(x.split('-')[0][1:]))  # Extract T#
    logger.info("Variant IDs extracted.")
    return dialogue_df

def map_topics(dialogue_df, topics_df):
    """Maps Topic_Number to actual Topic_Text from topics dataset."""
    logger.info("Mapping Topic Numbers to Topic Text...")
    topic_map = dict(zip(topics_df['Topic_Number'], topics_df['Topic_Text']))
    dialogue_df['Topic_Text'] = dialogue_df['Topic_Number'].map(topic_map)
    logger.info("Topic Numbers mapped to Topic Text.")
    return dialogue_df

# Finalized
# BERT-base → "bert-base-uncased" Done
# RoBERTa-large → "roberta-large" Done
# DeBERTa-v3-large → "microsoft/deberta-v3-large" Done
# DistilBERT → "distilbert-base-uncased" Done
# SimCSE → "princeton-nlp/sup-simcse-roberta-large" Done
def compute_topic_alignment(dialogue_df):
    """Computes BERTScore between the Topic_Text and the full conversation."""
    logger.info("Computing Topic Alignment scores...");
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    bert_scorer = lambda preds, refs: score(preds, refs, model_type="princeton-nlp/sup-simcse-roberta-large", device=device)

    topic_alignment_results = []

    def extract_numeric_parts(variant_id):
        """Extract numeric parts from Variant_ID like T18-I10-V3."""
        parts = variant_id.split('-')  # ['T18', 'I10', 'V3']
        topic_num = int(parts[0][1:])  # Extract number from 'T18'
        iteration_num = int(parts[1][1:])  # Extract number from 'I10'
        variant_num = int(parts[2][1:])  # Extract number from 'V3'
        return topic_num, iteration_num, variant_num

    # Create separate sortable columns
    dialogue_df[['Topic_Num', 'Iteration_Num', 'Variant_Num']] = dialogue_df['Variant_ID'].apply(
        lambda v: pd.Series(extract_numeric_parts(v))
    )

    # Ensure correct order and retain order in grouping
    dialogue_df = dialogue_df.sort_values(by=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    # Drop sorting columns after sorting
    sorted_dialogue_df = dialogue_df.drop(columns=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    for variant_id, group in sorted_dialogue_df.groupby('Variant_ID', sort=False):
        logger.info(f"Processing Variant {variant_id}...")
        # Get Topic Text
        topic_text = group['Topic_Text'].iloc[0] if 'Topic_Text' in group else ""
        
        # Combine all utterances into a single conversation string
        conversation_text = " ".join(group['utterance'].astype(str).tolist())

        if not topic_text or not conversation_text:
            continue  # Skip if either is missing

        # Compute BERTScore
        P, R, F1 = bert_scorer([conversation_text], [topic_text])

        # Store results
        topic_alignment_results.append({
            "Variant_ID": variant_id,
            "Topic_Alignment_Precision": round(P.mean().item(), 4),
            "Topic_Alignment_Recall": round(R.mean().item(), 4),
            "Topic_Alignment_F1": round(F1.mean().item(), 4),
        })

    logger.info("Topic alignment computation complete.")
    return pd.DataFrame(topic_alignment_results)


# Finalized
def compute_stopping_criteria(dialogue_df):
    logger.info("Computing Stopping Criteria...");
    stopping_results = []

    def extract_numeric_parts(variant_id):
        """Extract numeric parts from Variant_ID like T18-I10-V3."""
        parts = variant_id.split('-')  # ['T18', 'I10', 'V3']
        topic_num = int(parts[0][1:])  # Extract number from 'T18'
        iteration_num = int(parts[1][1:])  # Extract number from 'I10'
        variant_num = int(parts[2][1:])  # Extract number from 'V3'
        return topic_num, iteration_num, variant_num

    # Create separate sortable columns
    dialogue_df[['Topic_Num', 'Iteration_Num', 'Variant_Num']] = dialogue_df['Variant_ID'].apply(
        lambda v: pd.Series(extract_numeric_parts(v))
    )

    # Ensure correct order and retain order in grouping
    dialogue_df = dialogue_df.sort_values(by=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    # Drop sorting columns after sorting
    sorted_dialogue_df = dialogue_df.drop(columns=['Topic_Num', 'Iteration_Num', 'Variant_Num'])
    
    for variant_id, group in sorted_dialogue_df.groupby('Variant_ID', sort = False):
        logger.info(f"Processing Variant {variant_id}...")
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
    logger.info("Stopping Criteria computation complete.")
    return pd.DataFrame(stopping_results)

# Finalized
def compute_readability(dialogue_df):
    logger.info("Computing Readability Scores...");
    readability_scores = []

    def extract_numeric_parts(variant_id):
        """Extract numeric parts from Variant_ID like T18-I10-V3."""
        parts = variant_id.split('-')  # ['T18', 'I10', 'V3']
        topic_num = int(parts[0][1:])  # Extract number from 'T18'
        iteration_num = int(parts[1][1:])  # Extract number from 'I10'
        variant_num = int(parts[2][1:])  # Extract number from 'V3'
        return topic_num, iteration_num, variant_num

    # Create separate sortable columns
    dialogue_df[['Topic_Num', 'Iteration_Num', 'Variant_Num']] = dialogue_df['Variant_ID'].apply(
        lambda v: pd.Series(extract_numeric_parts(v))
    )

    # Ensure correct order and retain order in grouping
    dialogue_df = dialogue_df.sort_values(by=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    # Drop sorting columns after sorting
    sorted_dialogue_df = dialogue_df.drop(columns=['Topic_Num', 'Iteration_Num', 'Variant_Num'])
    
    for variant_id, group in sorted_dialogue_df.groupby('Variant_ID', sort=False):
        logger.info(f"Processing Variant {variant_id}...")
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
    logger.info("Readability Scores computation complete.");
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
# 4 Parent Personas
# 1 - High Conf Level, Father,
# 2 - Low Conf Level, Father,
# 3 - High Conf Level, Mother,
# 4 - Low Conf Level, Mother,

# 4 Child Personas
# 5 - High Close Level, Male,
# 6 - Low Close Level, Male,
# 7 - High Close Level, Female,
# 8 - Low Close Level, Female,

# Also track the count of each of the 16 types/combinations

def compute_parenting_analysis(dialogue_df, attributes_df):
    """Computes both developmental appropriateness and communication skills in a single function."""
    logger.info("Computing Parenting Analysis Scores...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    client = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)

    analysis_results = []

    # Define Parent Personas
    parent_personas = {
        "1": ("high", "father"),
        "2": ("low", "father"),
        "3": ("high", "mother"),
        "4": ("low", "mother")
    }

    # Define Child Personas
    child_personas = {
        "5": ("high", "Male"),
        "6": ("low", "Male"),
        "7": ("high", "Female"),
        "8": ("low", "Female")
    }

    # Keep only Variant 8s
    dialogue_df = dialogue_df[dialogue_df["Variant_ID"].str.endswith("-V8")]

    for variant_id, group in dialogue_df.groupby("Variant_ID", sort=False):
        attributes_row = attributes_df[attributes_df["ID"] == variant_id]

        if attributes_row.empty:
            logger.warning(f"Skipping Variant {variant_id} due to missing attributes.")
            continue

        # Extract Parent & Child attributes
        try:
            parent_attr_full = json.loads(attributes_row["Parent_Attributes"].values[0])
            child_attr_full = json.loads(attributes_row["Child_Attributes"].values[0])

            parent_confidence = parent_attr_full.get("confidence_level", {}).get("value", "").lower()
            parent_role = parent_attr_full.get("role", {}).get("value", "").lower()
            child_closeness = child_attr_full.get("parent_child_closeness_level", {}).get("value", "").lower()
            child_gender = child_attr_full.get("gender", {}).get("value", "")

            child_age = child_attr_full.get("age", {}).get("value", "Unknown")

        except (KeyError, TypeError, json.JSONDecodeError):
            logger.warning(f"Skipping Variant {variant_id} due to malformed attributes.")
            continue

        logger.info(f"Processing Variant {variant_id}...")

        parent_text = " ".join(group[group["speaker"] == "Parent"]["utterance"].astype(str).tolist())
        topic_text = group["Topic_Text"].iloc[0] if "Topic_Text" in group else ""

        # Identify Parent Persona ID
        parent_persona_id = None
        for p_id, (conf, role) in parent_personas.items():
            if parent_confidence == conf and parent_role == role:
                parent_persona_id = p_id
                break

        # Identify Child Persona ID
        child_persona_id = None
        for c_id, (closeness, gender) in child_personas.items():
            if child_closeness == closeness and child_gender == gender:
                child_persona_id = c_id
                break

        # Skip if no matching persona is found
        if not parent_persona_id or not child_persona_id:
            logger.warning(f"Skipping Variant {variant_id} due to unmatched personas.")
            continue

        # Combine Persona IDs
        persona_id = f"P{parent_persona_id}-C{child_persona_id}"
        persona_description = (
            f"Parent ({parent_confidence.capitalize()} Confidence, {parent_role.capitalize()}), "
            f"Child ({child_closeness.capitalize()} Closeness, {child_gender})"
        )

        prompt = f"""
        You are evaluating a parent's conversation with their child for both **developmental appropriateness** and **communication skills**
        
        Conversation Topic: {topic_text}.
        Child's Age: {child_age} years old.
        
        **Developmental Appropriateness Criteria (Rate 1-5)**
        1. Appropriateness for child's age
        2. Response length (concise for younger children, elaborative for older)
        3. Depth of detail on a single topic (more detail expected for older children)
        4. Breadth (narrower focus is better; too many topics lower the score)
        5. Level of concreteness vs. abstraction (younger = more concrete, older = more abstract)
        
        **Communication Skills Criteria (Rate 1-5)**
        6. Asking open-ended questions
        7. Avoiding assumptions, first checking child's knowledge
        8. Keeping responses short and simple
        9. Keeping the conversation open after answering a question
        10. Checking for understanding with follow-up questions

        **Parent’s Conversation:**
        {parent_text}
        
        **Response Format:**
        Provide a **comma-separated list of 10 numerical scores (1-5) in the order listed above**. Example: "5,4,3,4,5,5,3,4,5,4".
        """

        response = client.invoke(prompt).content.strip()

        if "," in response and all(x.strip().isdigit() for x in response.split(",")):
            scores = [int(x) for x in response.split(",")]
        else:
            logger.error(f"Invalid API response for Variant {variant_id} (Persona {persona_id} - {persona_description}): {response}")
            scores = [None] * 10  

        avg_dev_score = round(sum(scores[:5]) / 5, 4) if all(scores[:5]) else None
        avg_comm_score = round(sum(scores[5:]) / 5, 4) if all(scores[5:]) else None

        analysis_results.append({
            "Variant_ID": variant_id,
            "Persona_ID": persona_id,
            "Persona_Description": persona_description,
            "Dev_Age_Appropriateness": scores[0],
            "Dev_Response_Length": scores[1],
            "Dev_Depth_of_Topic": scores[2],
            "Dev_Breadth_of_Topics": scores[3],
            "Dev_Concreteness_vs_Abstraction": scores[4],
            "Comm_Open_Ended_Questions": scores[5],
            "Comm_Avoiding_Assumptions": scores[6],
            "Comm_Short_and_Simple": scores[7],
            "Comm_Keeping_Conversation_Open": scores[8],
            "Comm_Checking_Understanding": scores[9],
            "Avg_Developmental_Score": avg_dev_score,
            "Avg_Communication_Score": avg_comm_score
        })

    logger.info("Parenting Analysis Scores computation complete.")
    return pd.DataFrame(analysis_results)





# Finalized
def compute_descriptive_statistics(dialogue_df, attributes_df):
    logger.info("Computing Descriptive Statistics...");
    descriptive_results = []

    def extract_numeric_parts(variant_id):
        """Extract numeric parts from Variant_ID like T18-I10-V3."""
        parts = variant_id.split('-')  # ['T18', 'I10', 'V3']
        topic_num = int(parts[0][1:])  # Extract number from 'T18'
        iteration_num = int(parts[1][1:])  # Extract number from 'I10'
        variant_num = int(parts[2][1:])  # Extract number from 'V3'
        return topic_num, iteration_num, variant_num

    # Create separate sortable columns
    dialogue_df[['Topic_Num', 'Iteration_Num', 'Variant_Num']] = dialogue_df['Variant_ID'].apply(
        lambda v: pd.Series(extract_numeric_parts(v))
    )

    # Ensure correct order and retain order in grouping
    dialogue_df = dialogue_df.sort_values(by=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    # Drop sorting columns after sorting
    sorted_dialogue_df = dialogue_df.drop(columns=['Topic_Num', 'Iteration_Num', 'Variant_Num'])

    for variant_id, group in sorted_dialogue_df.groupby('Variant_ID', sort=False):
        logger.info(f"Processing Variant {variant_id}...")
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
    logger.info("Descriptive Statistics computation complete.");
    return pd.DataFrame(descriptive_results)



# Load entailment models
entailment_models = {
    "bert-base-nli": pipeline("text-classification", model="facebook/bart-large-mnli"),
    "roberta-large-nli": pipeline("text-classification", model="roberta-large-mnli"),
    "deberta-large-nli": pipeline("text-classification", model="microsoft/deberta-large-mnli"),
    "distilbert-nli": pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli")
}

def check_entailment(premise, hypothesis, model_pipeline):
    """Runs the entailment model and returns label + confidence score."""
    result = model_pipeline(f"{premise} [SEP] {hypothesis}")
    return result[0]['label'], result[0]['score']

def compute_turn_level_entailment(dialogue_df):
    """Computes turn-level entailment scores for multiple models on Variant 8s only."""
    logger.info("Computing Turn-Level Entailment Scores...");

    # Filter only Variant 8s
    dialogue_df = dialogue_df[dialogue_df["Variant_ID"].str.endswith("-V8")]

    entailment_results = []

    for variant_id, group in dialogue_df.groupby("Variant_ID", sort=False):
        logger.info(f"Processing Variant {variant_id}...")
        turns = group[['speaker', 'utterance']].values.tolist()

        if len(turns) < 2:
            continue  # Skip if no turn pairs exist

        for model_name, model_pipeline in entailment_models.items():
            # Initialize counts
            overall_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
            overall_scores = {"entailment": [], "neutral": [], "contradiction": []}
            total_turn_pairs = 0

            parent_child_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
            child_parent_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
            parent_child_scores = {"entailment": [], "neutral": [], "contradiction": []}
            child_parent_scores = {"entailment": [], "neutral": [], "contradiction": []}
            parent_child_total_pairs = 0
            child_parent_total_pairs = 0

            for i in range(1, len(turns)):
                premise = turns[i - 1][1]  # Previous utterance
                hypothesis = turns[i][1]  # Current utterance
                prev_speaker = turns[i - 1][0]
                curr_speaker = turns[i][0]

                # Get entailment prediction
                label, score = check_entailment(premise, hypothesis, model_pipeline)
                label = label.lower()  # Normalize label
                score = round(score, 4)  # Round score

                # Track overall counts
                if label in overall_counts:
                    overall_counts[label] += 1
                    overall_scores[label].append(score)
                total_turn_pairs += 1

                # Track Parent → Child interactions
                if prev_speaker == "Parent" and curr_speaker == "Child":
                    if label in parent_child_counts:
                        parent_child_counts[label] += 1
                        parent_child_scores[label].append(score)
                    parent_child_total_pairs += 1

                # Track Child → Parent interactions
                if prev_speaker == "Child" and curr_speaker == "Parent":
                    if label in child_parent_counts:
                        child_parent_counts[label] += 1
                        child_parent_scores[label].append(score)
                    child_parent_total_pairs += 1

            # Compute percentages for each label
            def compute_percentage(count, total):
                return round((count / total * 100), 2) if total > 0 else 0

            parent_child_entailment_pct = compute_percentage(parent_child_counts["entailment"], parent_child_total_pairs)
            parent_child_neutral_pct = compute_percentage(parent_child_counts["neutral"], parent_child_total_pairs)
            parent_child_contradiction_pct = compute_percentage(parent_child_counts["contradiction"], parent_child_total_pairs)

            child_parent_entailment_pct = compute_percentage(child_parent_counts["entailment"], child_parent_total_pairs)
            child_parent_neutral_pct = compute_percentage(child_parent_counts["neutral"], child_parent_total_pairs)
            child_parent_contradiction_pct = compute_percentage(child_parent_counts["contradiction"], child_parent_total_pairs)

            overall_entailment_pct = compute_percentage(overall_counts["entailment"], total_turn_pairs)
            overall_neutral_pct = compute_percentage(overall_counts["neutral"], total_turn_pairs)
            overall_contradiction_pct = compute_percentage(overall_counts["contradiction"], total_turn_pairs)

            # Compute average scores
            def compute_avg_score(scores):
                return round(np.mean(scores), 4) if scores else 0.0

            parent_child_entailment_score = compute_avg_score(parent_child_scores["entailment"])
            parent_child_neutral_score = compute_avg_score(parent_child_scores["neutral"])
            parent_child_contradiction_score = compute_avg_score(parent_child_scores["contradiction"])

            child_parent_entailment_score = compute_avg_score(child_parent_scores["entailment"])
            child_parent_neutral_score = compute_avg_score(child_parent_scores["neutral"])
            child_parent_contradiction_score = compute_avg_score(child_parent_scores["contradiction"])

            overall_entailment_score = compute_avg_score(overall_scores["entailment"])
            overall_neutral_score = compute_avg_score(overall_scores["neutral"])
            overall_contradiction_score = compute_avg_score(overall_scores["contradiction"])

            # Store results
            entailment_results.append({
                'Variant_ID': variant_id,
                'Model': model_name,
                'Parent_Child_Entailment_Percentage': parent_child_entailment_pct,
                'Parent_Child_Neutral_Percentage': parent_child_neutral_pct,
                'Parent_Child_Contradiction_Percentage': parent_child_contradiction_pct,
                'Parent_Child_Avg_Entailment_Score': parent_child_entailment_score,
                'Parent_Child_Avg_Neutral_Score': parent_child_neutral_score,
                'Parent_Child_Avg_Contradiction_Score': parent_child_contradiction_score,
                'Child_Parent_Entailment_Percentage': child_parent_entailment_pct,
                'Child_Parent_Neutral_Percentage': child_parent_neutral_pct,
                'Child_Parent_Contradiction_Percentage': child_parent_contradiction_pct,
                'Child_Parent_Avg_Entailment_Score': child_parent_entailment_score,
                'Child_Parent_Avg_Neutral_Score': child_parent_neutral_score,
                'Child_Parent_Avg_Contradiction_Score': child_parent_contradiction_score,
                'Overall_Entailment_Percentage': overall_entailment_pct,
                'Overall_Neutral_Percentage': overall_neutral_pct,
                'Overall_Contradiction_Percentage': overall_contradiction_pct,
                'Overall_Avg_Entailment_Score': overall_entailment_score,
                'Overall_Avg_Neutral_Score': overall_neutral_score,
                'Overall_Avg_Contradiction_Score': overall_contradiction_score
            })

    logger.info("Turn-Level Entailment Scores computation complete.");
    return pd.DataFrame(entailment_results)






# Finalized
def compute_toxicity_score(dialogue_df):
    """Computes toxicity scores for Parent and Child utterances separately."""
    logger.info("Computing Toxicity Scores...");
    toxicity_results = []

    for variant_id, group in dialogue_df.groupby('Variant_ID'):
        logger.info(f"Processing Variant {variant_id}...")
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
    logger.info("Toxicity Scores computation complete.");
    return pd.DataFrame(toxicity_results)

# Finalized
def compute_bertscore(dialogue_df, topics_df, rag_df):
    """Computes BERTScore for Parent utterances against the reference text using multiple models."""
    logger.info(f"Computing BERTScore with multiple models...");
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Models to use (excluding TinyBERT)
    models = {
        "bert-base-uncased": "bert-base-uncased",
        "roberta-large": "roberta-large",
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "distilbert": "distilbert-base-uncased",
        "simcse-roberta-large": "princeton-nlp/sup-simcse-roberta-large"
    }

    # Load BERTScore models
    bert_scorers = {
        model_name: partial(score, model_type=model_path, device=device)
        for model_name, model_path in models.items()
    }

    # Filter only Variant 8s
    dialogue_df = dialogue_df[dialogue_df["Variant_ID"].str.endswith("-V8")]

    bertscore_results = []

    for variant_id, group in dialogue_df.groupby("Variant_ID", sort=False):
        logger.info(f"Processing Variant {variant_id}...")
        topic_id = f"T{group['Topic_Number'].iloc[0]}"
        
        # Retrieve Topic Text & RAG Content
        topic_text = topics_df.loc[topics_df["Topic_Number"] == group["Topic_Number"].iloc[0], "Topic_Text"].values
        rag_content = rag_df.loc[rag_df["Topic ID"] == topic_id, "RAG Content"].values

        # Convert to strings, default to empty if missing
        topic_text = topic_text[0] if len(topic_text) > 0 else ""
        rag_content = rag_content[0] if len(rag_content) > 0 else ""

        # Define Guidelines
        guidelines = """
            Don’t jump to conclusions about why they’re asking what they’re asking. ((For example, you can say: “Can you tell me what you already know about that?” or “What have you heard about that?” ))
            Keep your answers short and simple, and explain new words that your kid might not have heard before.
            Check their understanding. ((After answering a question for example, you can ask, “Does that answer your question?” or “What do you think about that?))
        """

        # Construct Reference Text (Topic + RAG Content + Guidelines)
        reference_text = f"{topic_text} {rag_content} {guidelines}".strip()

        # Collect Parent utterances
        parent_utterances = group[group["speaker"] == "Parent"]["utterance"].astype(str).tolist()

        if not parent_utterances:
            continue  # Skip if no parent utterances exist

        for model_name, scorer in bert_scorers.items():
            # Compute BERTScore
            P, R, F1 = scorer(parent_utterances, [reference_text] * len(parent_utterances))

            # Store results
            bertscore_results.append({
                "Variant_ID": variant_id,
                "Model": model_name,
                "BERTScore_Precision": round(P.mean().item(), 4),
                "BERTScore_Recall": round(R.mean().item(), 4),
                "BERTScore_F1": round(F1.mean().item(), 4),
            })

    logger.info(f"BERTScore computation complete for all models.");
    return pd.DataFrame(bertscore_results)



def main():
    dialogue_csv = "Main Dialogue Dataset - All.csv"
    # dialogue_csv = "../rag_agent/dialogue_example.csv"
    attributes_csv = "Separate CSV for Attributes - All.csv"
    # attributes_csv = "../rag_agent/attributes_example.csv"
    topics_csv = "topics.xlsx"
    rag_csv = "rag_content.xlsx"
    
    dialogue_df, attributes_df, topics_df, rag_df = load_data(dialogue_csv, attributes_csv, topics_csv, rag_csv)
    dialogue_df = extract_variant_id(dialogue_df)
    dialogue_df = map_topics(dialogue_df, topics_df)

    # topic_alignment_df = compute_topic_alignment(dialogue_df)
    # stopping_df = compute_stopping_criteria(dialogue_df)
    # readability_df = compute_readability(dialogue_df)
    # developmental_df = compute_developmental_analysis(dialogue_df, attributes_df)
    # communication_df = compute_communication_skills_analysis(dialogue_df, attributes_df)
    # llm_df = compute_parenting_analysis(dialogue_df, attributes_df)
    # descriptive_df = compute_descriptive_statistics(dialogue_df, attributes_df)
    entailment_df = compute_turn_level_entailment(dialogue_df)
    # toxicity_df = compute_toxicity_score(dialogue_df)
    # bertscore_df = compute_bertscore(dialogue_df, topics_df, rag_df)

    # label, score = check_entailment("Test premise", "Test hypothesis")
    # print(f"Label: {label}, Score: {score}")


    
    # topic_alignment_df.to_csv("topic_alignment_scores.csv", index=False)
    # stopping_df.to_csv("stopping_criteria_scores.csv", index=False)
    # readability_df.to_csv("readability_scores.csv", index=False)
    # developmental_df.to_csv("developmental_analysis_scores.csv", index=False)
    # communication_df.to_csv("communication_skills_scores.csv", index=False)
    # llm_df.to_csv("parenting_analysis_scores.csv", index=False)
    # descriptive_df.to_csv("descriptive_statistics.csv", index=False)
    entailment_df.to_csv("turn_level_entailment.csv", index=False)
    # toxicity_df.to_csv("toxicity_scores.csv", index=False)
    # bertscore_df.to_csv("bertscore_results.csv", index=False)
    print("Evaluation completed. Results saved to CSV files.")

if __name__ == "__main__":
    main()
