import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, Document, load_index_from_storage, StorageContext
import textstat
import os
from dotenv import load_dotenv
import openai
import random

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_readable_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

def load_json_data(file_path):
    logging.info(f"Loading JSON data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    logging.info(f"Loaded {len(data)} records from JSON file")
    return data

def create_documents_from_posts(posts):
    logging.info("Creating documents from posts")
    documents = []
    for post in posts:
        doc_content = f"Title: {post['title']}\nContent: {post['content']}\nComment: {post['most_upvoted_comment']}\nURL: {post['url']}"
        documents.append(Document(text=doc_content))
    logging.info(f"Created {len(documents)} documents")
    return documents

def check_understanding(response: str, key_concepts: list) -> bool:
    """Check if the child mentions key concepts to ensure topic understanding."""
    response_lower = response.lower()
    for concept in key_concepts:
        if concept.lower() in response_lower:
            return True
    return False

def agents_agree_to_stop(parent_response: str, child_response: str) -> bool:
    """Check if both parent and child agree to stop the conversation."""
    stop_keywords = ["let’s stop", "end this", "we're done"]
    return any(keyword in parent_response.lower() for keyword in stop_keywords) and \
           any(keyword in child_response.lower() for keyword in stop_keywords)


def is_goal_met(response: str, goal_keywords: list) -> bool:
    """Check if the response aligns with the conversation goal."""
    response_lower = response.lower()
    return all(keyword.lower() in response_lower for keyword in goal_keywords)



def create_and_save_index(documents, index_file):
    if not os.path.exists(index_file):
        logging.info("Creating index from documents")
        index = GPTVectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_file)
        logging.info(f"Index saved to {index_file}")
    else:
        logging.info("Index already exists; no need to create a new one.")

def load_index(index_file):
    logging.info(f"Loading index from {index_file}")
    storage_context = StorageContext.from_defaults(persist_dir=index_file)
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded successfully")
    return index


sentiment_agent = ChatOpenAI(api_key=openai.api_key, model="gpt4-1106-preview", base_url=openai.api_base)

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment using OpenAI's ChatOpenAI agent and handle unexpected responses."""
    prompt = f"""
    You are a sentiment analysis tool. Analyze the sentiment of this text: "{text}".
    Return only one word: 'positive', 'neutral', or 'negative'.
    """
    try:
        response = sentiment_agent.invoke(prompt)
        sentiment = response.content.strip().lower()

        valid_sentiments = ['positive', 'neutral', 'negative']
        for word in sentiment.split(): 
            if word in valid_sentiments:
                return word

        logging.warning(f"Unexpected sentiment response: {sentiment}")
        return "unknown"

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return "unknown"


class ConversationCoach:
    def __init__(self, api_key: str, base_url: str, include_thought_process: bool = False, index=None):
        self.parent_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.child_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.chat_history = []
        self.parent_attributes = {}
        self.child_attributes = {}
        self.include_thought_process = include_thought_process
        self.index = index  
        self.disengagement_probability = 50

    def start_conversation(self, parent_attributes: dict, child_attributes: dict):
        self.parent_attributes = parent_attributes
        self.child_attributes = child_attributes
        self.conversation_id = generate_readable_id('conv')
        print(f"Starting Conversation ID: {self.conversation_id}\n")

    def retrieve_context(self, query_text: str) -> str:
        if self.index is None:
            logging.info("No index provided. Skipping context retrieval.")
            return ""  # Return an empty string if no context is used
        try:
            query_engine = self.index.as_query_engine()
            context = query_engine.query(query_text).response
            return context
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

    

    def generate_prompt(self, role: str, query: str, history: str = "", rag_context: str = ""):

        if role == "parent":
            if self.include_thought_process:
                prompt_template = """
                You are a parent with the following attributes: {parent_attributes}.
                Given your characteristics, you might approach this conversation in a certain way. Be mindful of your child's sensitivity and characteristics while responding.
                You are continuing a conversation with your child. So far, this has been said: {history}.
                Relevant context for this conversation is: {rag_context}.
                Your goal is to discuss the following topic with your child: {query}.
                Keep in mind that your child has the following characteristics: {child_attributes}.
                Make sure to respond with a simple vocabulary that suits the child's age.
                Respond concisely, empathetically, and continue the conversation if the child has just said something, adjust to your child's emotional state, and provide the thought process and analysis of what you are responding with.
                """
            else:
                prompt_template = """
                You are a parent with the following attributes: {parent_attributes}.
                You are continuing a conversation with your child. So far, this has been said: {history}.
                Relevant context for this conversation is: {rag_context}.
                Your goal is to discuss the following topic with your child: {query}.
                Keep in mind that your child has the following characteristics: {child_attributes}.
                Respond concisely and empathetically and continue the conversation if the child has just said something and adjust to your child's emotional state.
                Make sure to respond with a simple vocabulary that suits the child's age.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(parent_attributes=self.parent_attributes, query=query, child_attributes=self.child_attributes, history=history, rag_context=rag_context)

        elif role == "child":
            if self.include_thought_process:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                The conversation has a disengagement probability of {disengagement_probability}%.
                This means you may feel increasingly disengaged, frustrated, or bored as the conversation continues.
                If you feel disengaged or uninterested, integrate the word "stop" into your response.
                Your parent has said the following: {history}.
                Relevant context for this conversation is: {rag_context}.
                React concisely like a real teenager would based on your attributes, the nature of the conversation, and the disengagement probability.
                Use language and tone that reflect a real teenager's response: casual, emotional, and sometimes reactive.
                Make the teenager's response language that of Generation Alpha.
                """
            else:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                The conversation has a disengagement probability of {disengagement_probability}%.
                This means you may feel increasingly disengaged, frustrated, or bored as the conversation continues.
                If you feel disengaged or uninterested, integrate the word "stop" into your response.
                Your parent has said the following: {history}.
                Relevant context for this conversation is: {rag_context}.
                React concisely like a real teenager would based on your attributes, the nature of the conversation, and the disengagement probability.
                Use language and tone that reflect a real teenager's response: casual, emotional, and sometimes reactive.
                Make the teenager's response language that of Generation Alpha.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(child_attributes=self.child_attributes, history=history, rag_context=rag_context, disengagement_probability=self.disengagement_probability)


    def conduct_conversation(self, initial_query: str, output_file: str = "conversation_log.txt"):
        with open(output_file, "w") as file:
            parent_query = initial_query
            self.refusals = 0  # Track refusals
            key_concepts = ["consent", "protection", "disease", "pregnancy", "responsibility"]
            goal_keywords = ["safe sex", "consent", "protection"]

            file.write(f"Starting Conversation ID: {self.conversation_id}\n\n")

            for turn in range(10):  # Safeguard with a max turn limit
                # Retrieve RAG context
                rag_context = self.retrieve_context(parent_query)

                # Generate parent response
                history_summary = " ".join(self.chat_history[-2:])
                parent_prompt = self.generate_prompt("parent", parent_query, history=history_summary, rag_context=rag_context)
                parent_response = self.parent_agent.invoke(parent_prompt).content
                self.chat_history.append(f"Parent: {parent_response}")
                file.write(f"**** Parent:\n{parent_response}\n\n")

                # Generate child response
                history_summary = " ".join(self.chat_history[-2:])
                child_prompt = self.generate_prompt("child", parent_query, history=history_summary, rag_context=rag_context)
                child_response = self.child_agent.invoke(child_prompt).content
                self.chat_history.append(f"Child: {child_response}")
                file.write(f"**** Child:\n{child_response}\n\n")

                # Check disengagement probability and explicit refusal
                if "stop" in child_response.lower() in child_response.lower():
                    logging.info("Child explicitly expressed disengagement. Counting as a refusal.")
                    self.refusals += 1  # Count this as a refusal
                    if self.refusals >= 3:
                        logging.info("Stopping conversation due to three refusals from the child.")
                        file.write("\nConversation ended due to three refusals from the child.\n")
                        break

                # Check if conversation should stop
                sentiment = analyze_sentiment(child_response)
                # if check_understanding(child_response, key_concepts) and sentiment in ["positive"]:
                #     logging.info("Stopping conversation as the child understands the topic and shows positive sentiment.")
                #     file.write("\nConversation ended due to topic understanding and positive sentiment.\n")
                #     break
                if is_goal_met(child_response, goal_keywords):
                    logging.info("Stopping conversation as the child meets the conversation goal.")
                    file.write("\nConversation ended due to goal achievement.\n")
                    break
                if agents_agree_to_stop(parent_response, child_response):
                    logging.info("Stopping conversation as both agents agree to stop.")
                    file.write("\nConversation ended by mutual agreement.\n")
                    break

                # Increment disengagement probability
                self.disengagement_probability += random.randint(5, 10)  # Increase by 5-10%
                logging.info(f"Incremented disengagement probability to {self.disengagement_probability}%")

                # Prepare parent query for the next turn
                parent_query = f"Your child said: {child_response}. How do you respond?"

            file.write("\nConversation ended.\n")




    def end_conversation(self):
        print("\nConversation ended.")

def main():

    file_path = 'reddit_data.json' 
    posts_data = load_json_data(file_path)
    

    documents = create_documents_from_posts(posts_data)


    index_file = 'reddit_index'  
    create_and_save_index(documents, index_file)

    index = load_index(index_file)


    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    
    coach = ConversationCoach(api_key, base_url, include_thought_process=False, index=index)

    parent_attributes = {
        "confidence_level": {
            "value": "medium",
            "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with child."
        },
        "patience_level": {
            "value": "medium",
            "definition": "Ability to remain calm and composed during difficult and long conversations."
        },
        "triggers": {
            "value": ["disrespect"],
            "definition": "A trigger will cause the parent to react negatively and maybe even derail the conversation."
        },
        "comfort_level": {
            "value": "medium",
            "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality"
        },
        "open_dialogue": {
            "value": "medium",
            "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"
        }
    }
    child_attributes = {
        "age": 14,
        "temperament": "reactive",
        "openness_level": "low",
        "embarrassment_level" : "low",
        "trust_in_parent": "medium",
        "parent_child_closeness_level": "medium",
        "emotional_regulation": "low",
        "inclination to stay on topic (scale of 1-10)": "8",
        "triggers": ["mention of boyfriend coming over often"]
    }

    coach.start_conversation(parent_attributes, child_attributes)
    coach.conduct_conversation("You are turning 15 soon and I want to talk to you about safe sex.", output_file="conversation_output.txt")
    coach.end_conversation()

if __name__ == "__main__":
    main()
