import os
import json
import tiktoken
import openai

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")


def load_reddit_data(file_path: str):
    """Load posts from a JSON file that has a list of Reddit posts."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_documents_from_posts(posts):
    """
    Convert each post into a single Document object.
    Each post has fields - title, content, most_upvoted_comment, and url.
    """
    docs = []
    for post in posts:
        title = post.get("title", "")
        content = post.get("content", "")
        comment = post.get("most_upvoted_comment", "")
        url = post.get("url", "")

        combined_text = (
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Comment: {comment}\n"
            f"URL: {url}\n"
        )
        docs.append(Document(text=combined_text))
    return docs


def build_and_save_index(docs, index_dir="reddit_index"):
    """
    Create a token-aware VectorStoreIndex from the input documents and persist it.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")

    splitter = TokenTextSplitter(
        chunk_size=1024,
        chunk_overlap=20,
        separator=" ",
        tokenizer=encoding.encode
    )

    nodes = []
    for doc in docs:
        text_chunks = splitter.split_text(doc.text)
        for chunk in text_chunks:
            nodes.append(Document(text=chunk))

    llm = OpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=openai.api_key,
        openai_api_base=openai.base_url
    )
    embed_model = OpenAIEmbedding()

    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model, storage_context=storage_context)

    os.makedirs(index_dir, exist_ok=True)
    storage_context.persist(persist_dir=index_dir)

    print(f"Index persisted to '{index_dir}'.")
    return index


def load_existing_index(index_dir="reddit_index"):
    """
    Load a previously created VectorStoreIndex from disk.
    """
    if not os.path.exists(index_dir):
        print(f"Index directory '{index_dir}' does not exist. Please build the index first.")
        return None

    llm = OpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=openai.api_key,
        openai_api_base=openai.base_url
    )
    embed_model = OpenAIEmbedding()

    storage_context = StorageContext.from_defaults(persist_dir=index_dir)

    index = load_index_from_storage(
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model
    )
    print(f"Index loaded from '{index_dir}'.")
    return index


def main():
    file_path = "reddit_data.json"
    posts = load_reddit_data(file_path)
    print(f"Loaded {len(posts)} posts from '{file_path}'.")


    docs = create_documents_from_posts(posts)
    print(f"Created {len(docs)} Documents from the Reddit posts.")


    index_dir = "reddit_index"
    build_and_save_index(docs, index_dir=index_dir)

    loaded_index = load_existing_index(index_dir=index_dir)
    if loaded_index:
        query_engine = loaded_index.as_query_engine()
        response = query_engine.query("Puberty")
        print("Query Response:\n", response)


if __name__ == "__main__":
    main()
