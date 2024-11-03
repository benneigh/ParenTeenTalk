import json
import openai
import os
import logging
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, Document, load_index_from_storage, StorageContext


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
logging.info("Loaded environment variables from .env file")

# Set up your ChatGPT API key and base from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
logging.info("Configured OpenAI API key and base URL")

# Function to load JSON data from the file
def load_json_data(file_path):
    logging.info(f"Loading JSON data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    logging.info(f"Loaded {len(data)} records from JSON file")
    return data

# Function to create a document from the post
def create_documents_from_posts(posts):
    logging.info("Creating documents from posts")
    documents = []
    for post in posts:
        doc_content = f"Title: {post['title']}\nContent: {post['content']}\nComment: {post['most_upvoted_comment']}\nURL: {post['url']}"
        documents.append(Document(text=doc_content))
    logging.info(f"Created {len(documents)} documents")
    return documents

# Function to save the index to disk
def create_and_save_index(documents, index_file):
    if not os.path.exists(index_file):
        logging.info("Creating index from documents")
        index = GPTVectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_file)  # Persist index to disk
        logging.info(f"Index saved to {index_file}")
    else:
        logging.info("Index already exists; no need to create a new one.")

# Function to load the index from disk
def load_index(index_file):
    logging.info(f"Loading index from {index_file}")
    # Rebuild storage context from persisted directory
    storage_context = StorageContext.from_defaults(persist_dir=index_file)
    # Load index using the storage context
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded successfully")
    return index

# Function to query the index and get the relevant context
def query_index(index, query_text):
    logging.info(f"Querying index with text: {query_text}")
    context = index.query(query_text)
    logging.info("Query completed and context retrieved")
    return context

# Function to generate a response using the retrieved context and ChatGPT
def generate_response_from_context(context, query):
    logging.info("Generating response from context using ChatGPT")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information about teenage sexual health."},
            {"role": "user", "content": f"{context}\n\nQuery: {query}"}
        ]
    )
    answer = response['choices'][0]['message']['content']
    logging.info("Response generated successfully")
    return answer

# Main function to run the RAG-based system
def main():
    # Load the JSON file with Reddit posts
    file_path = 'reddit_data.json'  # Replace with your actual JSON file path
    posts_data = load_json_data(file_path)
    
    # Create documents from posts
    documents = create_documents_from_posts(posts_data)

    # Create and save the index
    index_file = 'reddit_posts_index'  # Use folder without extension
    create_and_save_index(documents, index_file)

    # Load the index
    index = load_index(index_file)

    # User query
    query = "Do I need to use protection every time?"

    # Query the index to get the relevant context
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    # Print the response
    logging.info("Final Response:")
    print("Response:", response)


if __name__ == "__main__":
    main()