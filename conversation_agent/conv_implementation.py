import argparse
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import textstat


def generate_readable_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

def word_count(text: str) -> int:
    return len(text.split())

def flesch_reading_ease(text: str) -> float:
    return textstat.flesch_reading_ease(text)

class ConversationCoach:
    def __init__(self, api_key: str, base_url: str, include_thought_process: bool = False): 
        

        self.parent_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.child_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)


        self.chat_history = []
        self.parent_attributes = {}
        self.child_attributes = {}
        self.include_thought_process = include_thought_process

    def start_conversation(self, parent_attributes: dict, child_attributes: dict):

        self.parent_attributes = parent_attributes
        self.child_attributes = child_attributes
        self.conversation_id = generate_readable_id('conv')
        print(f"Starting Conversation ID: {self.conversation_id}\n")

    def generate_prompt(self, role: str, query: str, history: str = ""):
        if role == "parent":
            if self.include_thought_process:
                prompt_template = """
                You are a parent with the following attributes: {parent_attributes}.
                Given your characteristics, you might approach this conversation in a certain way. Be mindful of your child's sensitivity and characteristics while responding.
                You are continuing a conversation with your child. So far, this has been said: {history}.
                Your goal is to discuss the following topic with your child: {query}.
                Keep in mind that your child has the following characteristics: {child_attributes}.
                Respond concisely, empathetically, and continue the conversation if the child has just said something and provide the thought process and analysis of what you are responding with.
                """
            else:
                prompt_template = """
                You are a parent with the following attributes: {parent_attributes}.
                You are continuing a conversation with your child. So far, this has been said: {history}.
                Your goal is to discuss the following topic with your child: {query}.
                Keep in mind that your child has the following characteristics: {child_attributes}.
                Respond concisely and empathetically and continue the conversation if the child has just said something.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(parent_attributes=self.parent_attributes, query=query, child_attributes=self.child_attributes, history=history)

        elif role == "child":
            if self.include_thought_process:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                Given your temperament and emotional state, you might feel certain ways about the current topic of conversation.
                Your parent has said the following: {history}.
                React concisely based on your attributes and the nature of the conversation and provide the thought process and analysis of what you are responding with.
                """
            else:
                prompt_template = """
                You are a child with the following characteristics: {child_attributes}.
                Your parent has said the following: {history}.
                React concisely based on your attributes and the nature of the conversation.
                Only provide the response and do not include the thought process behind it.
                """
            return prompt_template.format(child_attributes=self.child_attributes, history=history)
    
    def analyze_response(self, response: str, role: str):
        words = word_count(response)
        readability = flesch_reading_ease(response)
        print(f"{role} Response - Word Count: {words}, Readability (Flesch Reading Ease): {readability:.2f}\n\n\n")

    def conduct_conversation(self, initial_query: str):

        parent_query = initial_query
        for turn in range(3):  

            history_summary = " ".join(self.chat_history[-2:])
            parent_prompt = self.generate_prompt("parent", parent_query, history=history_summary)
            parent_response = self.parent_agent.invoke(parent_prompt).content


            print(f"**** Parent:\n {parent_response}")
            self.chat_history.append(f"Parent: {parent_response}")
            self.analyze_response(parent_response, "Parent")

            history_summary = " ".join(self.chat_history[-2:])

         
            child_prompt = self.generate_prompt("child", parent_query, history=history_summary)
            child_response = self.child_agent.invoke(child_prompt).content

            print(f"**** Child:\n {child_response}")
            self.chat_history.append(f"Child: {child_response}")
            self.analyze_response(child_response, "Child")

            parent_query = f"Your child said: {child_response}. How do you respond?"

            history_summary = " ".join(self.chat_history[-2:])

    def end_conversation(self):
        print("\nConversation ended.")
        

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description="Run a parent-child conversation simulation.")
    parser.add_argument('--include_thought_process', action='store_true', 
                        help="Include the thought process in conversation (default: False)")
    
    args = parser.parse_args()

    api_key = ""
    base_url = ""
    coach = ConversationCoach(api_key, base_url, include_thought_process=args.include_thought_process)

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

    coach.conduct_conversation("You're turning 15 soon. Let's talk about protection when it comes to sexual activities in your life.")

    coach.end_conversation()