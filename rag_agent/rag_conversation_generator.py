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


class ConversationCoach:
    def __init__(self, api_key: str, base_url: str, include_thought_process: bool = False, index=None):
        self.parent_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.child_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.chat_history = []
        self.parent_attributes = {}
        self.child_attributes = {}
        self.include_thought_process = include_thought_process
        self.index = index  

    def start_conversation(self, parent_attributes: dict, child_attributes: dict):
        self.parent_attributes = parent_attributes
        self.child_attributes = child_attributes
        self.conversation_id = generate_readable_id('conv')
        print(f"Starting Conversation ID: {self.conversation_id}\n")

    def retrieve_context(self, query_text: str) -> str:
        query_engine = self.index.as_query_engine()
        context = query_engine.query(query_text).response 
        return context

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
                Respond concisely, empathetically, and continue the conversation if the child has just said something and provide the thought process and analysis of what you are responding with.
                """
            else:
                prompt_template = """
                You are a parent with the following attributes: {parent_attributes}.
                You are continuing a conversation with your child. So far, this has been said: {history}.
                Relevant context for this conversation is: {rag_context}.
                Your goal is to discuss the following topic with your child: {query}.
                Keep in mind that your child has the following characteristics: {child_attributes}.
                Respond concisely and empathetically and continue the conversation if the child has just said something.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(parent_attributes=self.parent_attributes, query=query, child_attributes=self.child_attributes, history=history, rag_context=rag_context)

        elif role == "child":
            if self.include_thought_process:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                Given your temperament and emotional state, you might feel certain ways about the current topic of conversation.
                Your parent has said the following: {history}.
                Relevant context for this conversation is: {rag_context}.
                React concisely based on your attributes and the nature of the conversation and provide the thought process and analysis of what you are responding with.
                """
            else:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                Your parent has said the following: {history}.
                Relevant context for this conversation is: {rag_context}.
                React concisely based on your attributes and the nature of the conversation.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(child_attributes=self.child_attributes, history=history, rag_context=rag_context)


    def conduct_conversation(self, initial_query: str, output_file: str = "conversation_log.txt"):
   
        with open(output_file, "w") as file:

            parent_query = initial_query
            file.write(f"Starting Conversation ID: {self.conversation_id}\n\n")
            
            for turn in range(3): 

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

                parent_query = f"Your child said: {child_response}. How do you respond?"

            file.write("\nConversation ended.\n")


    def end_conversation(self):
        print("\nConversation ended.")

def main():

    file_path = 'r_teenagers.json' 
    posts_data = load_json_data(file_path)
    

    documents = create_documents_from_posts(posts_data)


    index_file = 'reddit_posts_index'  
    create_and_save_index(documents, index_file)

    index = load_index(index_file)


    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    
    coach = ConversationCoach(api_key, base_url, include_thought_process=False, index=index)

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
            "neuroticism": "high"
        }
    }
    child_attributes = {
        "age": 14,
        "temperament": "reactive",
        "openness_level": "high",
        "trust_in_parent": "medium",
        "emotional_regulation": "low",
        "inclination to stay on topic (scale of 1-10)": "8",
        "triggers": ["mention of boyfriend coming over often"]
    }

    coach.start_conversation(parent_attributes, child_attributes)
    coach.conduct_conversation("You are turning 15 soon and I want to talk to you about safe sex.", output_file="conversation_output.txt")
    coach.end_conversation()

if __name__ == "__main__":
    main()
