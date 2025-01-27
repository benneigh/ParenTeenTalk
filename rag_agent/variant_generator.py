import os
import logging
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.core import load_index_from_storage, StorageContext

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VariantGenerator:
    def __init__(self, api_key: str, base_url: str, index=None):
        self.api_key = api_key
        self.base_url = base_url
        self.index = index
        # Change it to 4-o
        self.parent_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.child_agent = ChatOpenAI(api_key=api_key, model="gpt4-1106-preview", base_url=base_url)
        self.disengagement_probability = 50
        self.refusals = 0

    def retrieve_context(self, query_text: str) -> str:
        if self.index is None:
            logging.info("No index provided. Skipping context retrieval.")
            return ""
        try:
            query_engine = self.index.as_query_engine()
            context = query_engine.query(query_text).response
            return context
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

    def generate_prompt(self, role, query, guidelines=None, user_profiles=None, context="", history="", disengagement_probability=10):
        if user_profiles:
            child_attributes = """age: 14,
                embarrassment_level: low,
                parent_child_closeness_level: medium"""
        else:
            child_attributes = "Not specified"

        if role == "parent":
            prompt_template = """
            You are a parent with the following attributes:
            {parent_attributes}.
            Each attribute is defined as follows:
            {attribute_definitions}.
            You are continuing a conversation with your child. So far, this has been said: {history}.
            Relevant context for this conversation is: {rag_context}.
            Your goal is to discuss the following topic with your child: {query}.
            Keep in mind that your child has the following characteristics: {child_attributes}.
            Keep your response realistic, natural, and reflective of how a parent would actually speak to their child.
            Circle back to the main topic whenever the conversation drifts off course, but limit this to only three times during the discussion.
            Only provide the response and do not include your thought process behind it.
            """
        elif role == "child":
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
            Only provide the response and do not include your thought process behind it.
            """

        # Build attribute strings
        attribute_definitions = "\n".join(
            f"- {key}: {attr['definition']}" for key, attr in (user_profiles or {}).items()
        )
        parent_attributes = ", ".join(
            f"{key}: {attr['value']}" for key, attr in (user_profiles or {}).items()
        )

        return prompt_template.format(
            parent_attributes=parent_attributes or "Not specified",
            attribute_definitions=attribute_definitions or "Not specified",
            query=query,
            child_attributes=child_attributes,
            history=history,
            rag_context=context,
            disengagement_probability=disengagement_probability
        )


    def agents_agree_to_stop(self, parent_response: str, child_response: str) -> bool:
        stop_keywords = ["let’s stop", "end this", "we're done"]
        return any(keyword in parent_response.lower() for keyword in stop_keywords) and \
               any(keyword in child_response.lower() for keyword in stop_keywords)

    def is_goal_met(self, response: str, goal_keywords: list) -> bool:
        response_lower = response.lower()
        return all(keyword.lower() in response_lower for keyword in goal_keywords)

    def run_variant(self, query, variant_config):
        context = self.retrieve_context(query) if variant_config.get("context") else ""
        guidelines = variant_config.get("guidelines")
        user_profiles = variant_config.get("user_profiles")

        parent_history = ""
        child_history = ""
        goal_keywords = ["safe sex", "consent", "protection"]
        self.refusals = 0  # Reset refusal count for each variant

        with open("variant_results.txt", "a") as file:  # Append to file for all variants
            file.write(f"Starting Variant: {variant_config}\n")

            for turn in range(10):  # Max turn limit
                # Generate parent response
                parent_prompt = self.generate_prompt("parent", query, guidelines, user_profiles, context, child_history)
                try:
                    parent_response = self.parent_agent.invoke(parent_prompt).content.strip()
                except Exception as e:
                    logging.error(f"Error generating parent response: {e}")
                    parent_response = "Error"

                # Write parent response immediately to the file
                file.write(f"**** Parent:\n{parent_response}\n\n")

                # Update parent history
                parent_history += f"Parent: {parent_response}\n"

                # Generate child response
                child_prompt = self.generate_prompt("child", query, guidelines, user_profiles, context, parent_history, disengagement_probability=self.disengagement_probability)
                try:
                    child_response = self.child_agent.invoke(child_prompt).content.strip()
                except Exception as e:
                    logging.error(f"Error generating child response: {e}")
                    child_response = "Error"

                # Write child response immediately to the file
                file.write(f"**** Child:\n{child_response}\n\n")

                # Update child history
                child_history += f"Child: {child_response}\n"

                # Check stopping conditions
                if "stop" in child_response.lower():
                    self.refusals += 1
                    logging.info(f"Child disengaged. Refusals count: {self.refusals}")
                    if self.refusals >= 3:
                        file.write("\nConversation ended due to three refusals from the child.\n")
                        logging.info("Stopping conversation due to three refusals from the child.")
                        break

                if self.is_goal_met(child_response, goal_keywords):
                    file.write("\nConversation ended due to goal achievement.\n")
                    logging.info("Goal achieved. Stopping conversation.")
                    break

                if self.agents_agree_to_stop(parent_response, child_response):
                    file.write("\nConversation ended by mutual agreement.\n")
                    logging.info("Both agents agreed to stop the conversation.")
                    break

                # Increment disengagement probability
                self.disengagement_probability += random.randint(5, 10)
                logging.info(f"Incremented disengagement probability to {self.disengagement_probability}%")

            file.write("\nConversation ended.\n\n")

        return parent_history + child_history



    def execute_variants(self, query, variants, output_file="variant_results.txt"):
        results = {}
        with open(output_file, "w") as file:  # Open in write mode to clear previous content
            for variant_name, config in variants.items():
                
                # Reset refusal count before starting a new variant
                self.refusals = 0  
                self.disengagement_probability = 50  # Reset disengagement probability as well
            
                interaction = self.run_variant(query, config)
                results[variant_name] = interaction

        return results


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    index_file = 'reddit_index'
    index = None
    if os.path.exists(index_file):
        storage_context = StorageContext.from_defaults(persist_dir=index_file)
        index = load_index_from_storage(storage_context)

    generator = VariantGenerator(api_key, base_url, index=index)

    variants = {
        "Variant 1 - Basic": {"context": False, "guidelines": None, "user_profiles": None},
        "Variant 2 - Query + Reddit Context": {"context": True, "guidelines": None, "user_profiles": None},
        "Variant 3 - Query + Guidelines": {
            "context": False,
            "guidelines": "Respond concisely, empathetically, and adhere to constraints.",
            "user_profiles": None
        },
        "Variant 4 - Query + User Profiles": {
            "context": False,
            "guidelines": None,
            "user_profiles": {
                "confidence_level": {"value": "medium", "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with child."},
                "triggers": {"value": ["disrespect"], "definition": "A trigger will cause the parent to react negatively and maybe even derail the conversation."},
                "comfort_level": {"value": "medium", "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality"},
                "open_dialogue": {"value": "medium", "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"}
            }
        },
        "Variant 5 - Query + Context + Guidelines": {
            "context": True,
            "guidelines": "Respond concisely, empathetically, and adhere to constraints.",
            "user_profiles": None
        },
        "Variant 6 - Query + Context + User Profiles": {
            "context": True,
            "guidelines": None,
            "user_profiles": {
                "confidence_level": {"value": "medium", "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with child."},
                "triggers": {"value": ["disrespect"], "definition": "A trigger will cause the parent to react negatively and maybe even derail the conversation."},
                "comfort_level": {"value": "medium", "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality"},
                "open_dialogue": {"value": "medium", "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"}
            }
        },
        "Variant 7 - Query + Guidelines + User Profiles": {
            "context": False,
            "guidelines": "Respond concisely, empathetically, and adhere to constraints.",
            "user_profiles": {
                "confidence_level": {"value": "medium", "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with child."},
                "triggers": {"value": ["disrespect"], "definition": "A trigger will cause the parent to react negatively and maybe even derail the conversation."},
                "comfort_level": {"value": "medium", "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality"},
                "open_dialogue": {"value": "medium", "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"}
            }
        },
        "Variant 8 - Query + Context + Guidelines + User Profiles": {
            "context": True,
            "guidelines": "Respond concisely, empathetically, and adhere to constraints.",
            "user_profiles": {
                "confidence_level": {"value": "medium", "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with child."},
                "triggers": {"value": ["disrespect"], "definition": "A trigger will cause the parent to react negatively and maybe even derail the conversation."},
                "comfort_level": {"value": "medium", "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality"},
                "open_dialogue": {"value": "medium", "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"}
            }
        }
    }

    query = "You are turning 15 soon, and I want to talk to you about safe sex."
    output_file = "variant_results.txt"
    generator.execute_variants(query, variants, output_file=output_file)

    logging.info("Variant generation completed. Results saved to file.")

if __name__ == "__main__":
    main()
