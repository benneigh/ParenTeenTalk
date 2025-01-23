import os
from concurrent.futures import ThreadPoolExecutor
from advanced_conv_generator import *


def generate_conversation(model_name, conversation_id, output_dir="conversations", index=None):
    """
    Generate a single conversation based on the model configuration.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Set up attributes based on model
    parent_attributes = {}
    child_attributes = {}
    include_thought_process = False
    use_context = False

    if model_name == "baseline":
        # No additional attributes or context
        parent_attributes = {}
        child_attributes = {}
    elif model_name == "prompt_engineered":
        # Add prompt-engineered attributes
        parent_attributes = {
            "confidence_level": "high",
            "vocab_complexity": "simple",
            "patience_level": "medium",
            "triggers": ["disrespect"],
            "parenting_style": "strict",
        }
        child_attributes = {
            "age": 14,
            "temperament": "reactive",
            "openness_level": "high",
            "trust_in_parent": "medium",
            "emotional_regulation": "low",
        }
    elif model_name == "context_and_full":
        # Full context and attributes
        parent_attributes = {
            "confidence_level": "high",
            "vocab_complexity": "simple",
            "patience_level": "medium",
            "triggers": ["disrespect"],
            "parenting_style": "strict",
            "big_five": {
                "openness": "high",
                "conscientiousness": "medium",
                "extraversion": "low",
                "agreeableness": "medium",
                "neuroticism": "high",
            },
        }
        child_attributes = {
            "age": 14,
            "temperament": "reactive",
            "openness_level": "high",
            "trust_in_parent": "medium",
            "emotional_regulation": "low",
            "inclination to stay on topic (scale of 1-10)": "8",
            "triggers": ["mention of boyfriend coming over often"],
        }
        include_thought_process = False
        use_context = True

    # Initialize ConversationCoach
    coach = ConversationCoach(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        include_thought_process=include_thought_process,
        index=index if use_context else None,  # Pass index only for models using context
    )

    # Check if context should be used
    output_file = os.path.join(output_dir, f"{model_name}_conversation_{conversation_id}.txt")
    try:
        coach.start_conversation(parent_attributes, child_attributes)
        coach.conduct_conversation(
            "You are turning 15 soon and I want to talk to you about safe sex.",
            output_file
        )
    except AttributeError as e:
        logging.error(f"Error during conversation generation for {model_name}: {e}")
    finally:
        coach.end_conversation()





def batch_process_conversations(models, num_conversations, index, output_dir="conversations"):
    os.makedirs(output_dir, exist_ok=True)

    # Run in parallel for all models
    with ThreadPoolExecutor(max_workers=10) as executor:
        for model in models:
            model_output_dir = os.path.join(output_dir, model)
            futures = [
                executor.submit(generate_conversation, model, i, model_output_dir, index)
                for i in range(num_conversations)
            ]
            for future in futures:
                future.result()  # Wait for each task to complete



def main():
    # Path to the index file
    index_file = 'reddit_posts_index'

    # Check if the index file exists
    if os.path.exists(index_file):
        logging.info(f"Index file {index_file} exists. Loading it.")
        index = load_index(index_file)  # Load existing index
    else:
        logging.info(f"Index file {index_file} does not exist. Creating a new one.")
        # Load data and create documents
        file_path = 'r_teenagers.json'
        posts_data = load_json_data(file_path)

        documents = create_documents_from_posts(posts_data)

        # Create and save index
        create_and_save_index(documents, index_file)
        index = load_index(index_file)  # Load the newly created index

    # Define models and number of conversations
    models = ["baseline", "prompt_engineered", "context_and_full"]
    num_conversations = 1000  # Number of conversations per model

    # Batch process conversations for all models
    batch_process_conversations(models, num_conversations, index=index)





if __name__ == "__main__":
    main()
