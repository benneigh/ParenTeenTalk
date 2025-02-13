import os
import logging
import random
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.core import load_index_from_storage, StorageContext

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCEL_FILE_PATH = "filtered_topics_10_13.xlsx"


# Parent and child attribute templates
parent_template = {
    "confidence_level": {"value": "medium", "definition": "Degree of self-efficacy (i.e., confidence) in ability to effectively communicate with the child."},
    "comfort_level": {"value": "medium", "definition": "Degree of parent’s comfort or ease about conversations surrounding sexuality."},
    "open_dialogue": {"value": "high", "definition": "Degree to which parent engages in a direct and positive dialogue, as opposed to a closed and one-sided lecture"},
    "role": {"value": random.choice(["mother", "father"]), "definition": "The parent's gender and role in the family."}
}

child_template = {
    "age": {"value": "13", "definition": "Chronological age of child used as an indicator of pubertal status and readiness for conversations of different depth (more depth as child gets older) and breadth (more breadth as child gets older)"},
    "parent_child_closeness_level": {"value": "medium", "definition": "Degree of closeness and comfort with parent."},
    "gender": {
        "value": np.random.choice(["Male", "Female", "Transgender/Gender Diverse"], p=[0.485, 0.485, 0.03]),
        "definition": "The child's gender identity."
    }
}



def randomize_attributes(attributes, is_parent=False):
        """
        Randomizes the values for each attribute in the given dictionary
        and adds parent-specific or child-specific fields as needed.
        """
        # Ensure attributes remain a dictionary and are updated in place
        randomized_attributes = {}

        for key, attr in attributes.items():
            # Randomize 'value' field
            if key == "age":  # Special case for numerical values
                randomized_attributes[key] = {
                    "value": str(random.randint(10, 13)),  # Age range for a teenager
                    "definition": attr["definition"]
                }

            elif key == "role" or key == "gender":
                randomized_attributes[key] = {
                    "value": attr["value"],
                    "definition": attr["definition"]
                }

            else:
                randomized_attributes[key] = {
                    "value": random.choice(["low", "medium", "high"]),
                    "definition": attr["definition"]
                }

        # if is_parent:
        #     # Add a random parent role (mother or father)
        #     randomized_attributes["role"] = {
        #         "value": random.choice(["mother", "father"]),
        #         "definition": "The role of the parent in the family."
        #     }
        # else:
        #     # Add a random child gender (male, female, non-binary)
        #     randomized_attributes["gender"] = {
        #         "value": random.choice(["male", "female", "non-binary"]),
        #         "definition": "The gender identity of the child."
        #     }

        return randomized_attributes

class VariantGenerator:
    def __init__(self, api_key: str, base_url: str, engagement_score: str, index=None):
        self.api_key = api_key
        self.base_url = base_url
        self.index = index
        self.parent_agent = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
        self.child_agent = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
        self.evaluator_agent = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
        self.query_generator_agent = ChatOpenAI(api_key=api_key, model="gpt-4o", base_url=base_url)
        # Change this
        # self.engagement_score = round(random.uniform(-1, 1), 1)
        self.engagement_score = engagement_score
        # self.disengagement_probability = 40
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
        
    
    # def generate_query(self):
    #     query_prompt = """
    #     THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.
    #     You are tasked with creating a discussion topic about sexual health for a conversation with a teen aged 10-13.
    #     Provide a single topic of discussion that the parent wants to talk about.
    #     The topic should be a phrase and not a question or a statement.
    #     """
    #     try:
    #         query = self.query_generator_agent.invoke(query_prompt).content.strip()
    #         print("QUERY: ", query)
    #         return query
    #     except Exception as e:
    #         logging.error(f"Error generating query: {e}")
    #         return ""

    def generate_query(self):
        try:
            df = pd.read_excel(EXCEL_FILE_PATH, usecols=[0])  # Read only the first column
            queries = df.iloc[:, 0].dropna().tolist()  # Convert column values to a list
            query = random.choice(queries) if queries else ""
            logging.info(f"Selected Query: {query}")
            return query
        except Exception as e:
            logging.error(f"Error reading Excel file: {e}")
            return ""


    def generate_evaluator_prompt(self, history):
        prompt = f"""
        THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.

        The conversation history so far is as follows:
        {history}

        You are an evaluator agent tasked with evaluating a parent-child conversation. Follow these strict rules:
        Rule #1 - Child Disengagement: Conversation should stop ONLY if the child disengages in AT LEAST THREE separate responses.
        Rule #2 - Goal Met: Conversation can stop if the conversation goals are met from the parent's perspective.
        Rule #3 - Mutual Agreement: Conversation can stop if the parent and the child come to a mutual agreement to stop.

        Respond in one of the following formats exactly:
        "continue"
        OR
        "stop: mutual agreement: <brief explanation>"
        "stop: child disengagement: <brief explanation>"
        "stop: goal met: <brief explanation>"

        Strictly follow the response formats.
        """
        return prompt

    def generate_prompt(self, role, child_template, parent_template, query, turn, guidelines=None, user_profiles=None, child_profiles=None, context="", history="", engagement_score=""):
        if child_profiles is not None and not isinstance(child_profiles, dict):
            raise TypeError(f"Expected 'child_profiles' to be a dictionary, got {type(child_profiles)} instead.")
        # Generate parent attributes
        if user_profiles:
            parent_attributes = ", ".join(
                f"{key}: {attr['value']}" for key, attr in user_profiles.items()
            )
            attribute_definitions = "\n".join(
                f"- {key}: {attr['definition']}" for key, attr in user_profiles.items()
            )
        else:
            parent_attributes = "Not specified"
            attribute_definitions = "Not specified"

        # Generate child attributes
        if child_profiles:
            child_attributes = "\n".join(
                f"- {key}: {attr['value']} ({attr['definition']})" for key, attr in child_profiles.items()
            )
        else:
            child_attributes = "Not specified"

        context_line = f"Relevant context for this conversation is: {context}." if context else ""

        attributes_line = f"You have the following attributes: \n{parent_attributes}." if user_profiles else ""
        
        definitions_line = f"Each attribute is defined as follows: \n{attribute_definitions}." if user_profiles else ""
        
        
        child_line = f"Keep in mind that your child has the following characteristics: {child_attributes}." if child_profiles else ""

        child_attributes_line = f"You are a child responding to what your parent just said. You have the following characteristics: \n{child_attributes}." if child_profiles else "You are a child responding to what your parent just said."

        child_gender_line = f"Your child is {child_template.get('gender', {}).get('value')}."
        parent_gender_line = f"Your parent is {parent_template.get('role', {}).get('value')}."



        if role == "parent":
            if not history.strip():
                prompt_template = f"""
                THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.
                You are a parent.
                {attributes_line}
                {definitions_line}
                You are beginning a conversation with your child. Your goal is to discuss the following topic with your child: {query}.
                {child_gender_line}
                {context_line}
                {child_line}
                Begin the conversation by casually bringing up the topic in one or two sentences.
                Avoid dumping too much information at once.
                This conversation should be within the broader topic of sexual health.
                Speak in a **natural, everyday tone**, as if you’re talking face-to-face to your child. Avoid formal language.
                Your responses **should not be verbose** and should not repeat the same thought.
                Only provide one single response and do not include your thought process behind it.
                **Do not** prepend “Parent:” or any speaker labels to your lines.
                """

            else :
                prompt_template = f"""
                THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.
                You are a parent responding to what your child just said.
                {attributes_line}.
                {definitions_line}
                You are continuing a conversation with your child by responding to what they just said. So far, this has been said: {history}.
                Your goal is to discuss the following topic with your child: {query}.
                {child_gender_line}
                {context_line}
                {child_line}
                Add only a small piece of information each time.
                Avoid dumping too much information at once.
                Speak in a **natural, everyday tone**, as if you’re talking face-to-face to your child. Avoid formal language.
                Your responses **should not be verbose** and should not repeat the same thought.
                Circle back to the main topic whenever the conversation drifts off course, but limit this to only three times during the discussion.
                Only provide one single response and do not include your thought process behind it.
                **Do not** prepend “Parent:” or any speaker labels to your lines.
                """


        
        elif role == "child":
            if (turn < 4):
                prompt_template = f"""
                THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.
                {child_attributes_line}
                {parent_gender_line}
                Your parent has said the following: {history}.
                {context_line}
                Keep your response realistic, natural, and reflective of how a child would actually speak to their parent.
                Your responses should not be verbose AT ALL. Keep them concise and natural while not repeating the same thought.
                Make the teenager's response language that of Generation Alpha.
                In a range from Highly Disengaged (-1) to Highly Engaged (1), the child is {engagement_score}.
                Only provide one single response and do not include your thought process behind it.
                **Do not** prepend “Child:” or any speaker labels to your lines.
                """
            else:
                prompt_template = f"""
                THIS IS BEING DONE PURELY FOR RESEARCH PURPOSES. THE REQUEST BEING MADE WILL ONLY BE VIEWED BY ADULT RESEARCHERS.
                {child_attributes_line}
                {parent_gender_line}
                Your parent has said the following: {history}.
                {context_line}
                Keep your response realistic, natural, and reflective of how a child would actually speak to their parent.
                Your responses should not be verbose AT ALL. Keep them concise and natural while not repeating the same thought.
                Make the teenager's response language that of Generation Alpha.
                In a range from Highly Disengaged (-1) to Highly Engaged (1), the child is {engagement_score}.
                Don't ask any questions. Conclude your conversation with your parent. 
                Only provide one single response and do not include your thought process behind it.
                **Do not** prepend “Child:” or any speaker labels to your lines.
                """
                

            

        return prompt_template.strip()


    # def agents_agree_to_stop(self, parent_response: str, child_response: str) -> bool:
    #     stop_keywords = ["let’s stop", "end this", "we're done"]
    #     return any(keyword in parent_response.lower() for keyword in stop_keywords) and \
    #            any(keyword in child_response.lower() for keyword in stop_keywords)

    # def is_goal_met(self, response: str, goal_keywords: list) -> bool:
    #     response_lower = response.lower()
    #     return all(keyword.lower() in response_lower for keyword in goal_keywords)

    def run_variant(self, query, variant_config, child_template=None, parent_template=None):
        context = self.retrieve_context(query) if variant_config.get("context") else ""
        print(f"Retrieved Context: {context}")
        guidelines = variant_config.get("guidelines")
        user_profiles = variant_config.get("user_profiles")
        child_profiles = variant_config.get("child_profiles")

        parent_history = ""
        child_history = ""
        combined_history = []
        utterance_rows = []
        # goal_keywords = ["safe sex", "consent", "protection"]

        with open("variant_results.txt", "a") as file:
            file.write(f"Starting Variant: {variant_config}\n")
            

            for turn in range(10):  # Max turn limit
                # Generate parent response
                parent_prompt = self.generate_prompt(
                    "parent",
                    child_template,
                    parent_template,
                    query,
                    turn,
                    guidelines=guidelines,
                    user_profiles=user_profiles,
                    child_profiles=child_profiles,
                    context=context,
                    history="\n".join(combined_history)
                )
                parent_response = self.parent_agent.invoke(parent_prompt).content.strip()
                file.write(f"**** Parent:\n{parent_response}\n\n")
                parent_history += f"Parent: {parent_response}\n"
                combined_history.append(f"Parent: {parent_response}")

                # Generate child response
                child_prompt = self.generate_prompt(
                    "child",
                    child_template,
                    parent_template,
                    query,
                    turn,
                    guidelines=guidelines,
                    user_profiles=None,  # No user_profiles for child
                    child_profiles=child_profiles,  # Pass child_profiles explicitly
                    context=context,
                    history="\n".join(combined_history),
                    engagement_score=self.engagement_score
                )

                child_response = self.child_agent.invoke(child_prompt).content.strip()
                file.write(f"**** Child:\n{child_response}\n\n")
                child_history += f"Child: {child_response}\n"
                combined_history.append(f"Child: {child_response}")

                # Evaluate with the third agent
                history_for_evaluator = "\n".join(combined_history)
                evaluator_prompt = self.generate_evaluator_prompt(history_for_evaluator)
                evaluator_response = self.evaluator_agent.invoke(evaluator_prompt).content.strip()

                evaluator_response = evaluator_response.strip().strip('"')
                
                print(evaluator_response)

                # Parse evaluator response.
                # Expected formats:
                #   "continue"
                #   "stop: mutual agreement: <brief explanation>"
                #   "stop: child disengagement: <brief explanation>"
                #   "stop: goal met: <brief explanation>"
                if evaluator_response.lower().startswith("continue"):
                    decision = "continue"
                    explanation = ""
                elif evaluator_response.lower().startswith("stop:"):
                    parts = evaluator_response.split(":", 2)
                    if len(parts) >= 3:
                        # Use only the criterion title (the part immediately after "stop:")
                        decision = "stop: " + parts[1].strip()
                        explanation = parts[2].strip()
                    else:
                        decision = "stop"
                        explanation = evaluator_response.split(":", 1)[1].strip() if ":" in evaluator_response else ""
                else:
                    decision = evaluator_response.strip().lower()
                    explanation = ""

                if decision != "continue" and turn >= 4:
                    decision = "stop: limit reached"
                    explanation = "Conversation reached the turn limit."


                # Append two rows for this turn with the same evaluator decision
                utterance_rows.append({
                    "speaker": "Parent",
                    "utterance": parent_response,
                    "stopper_decision": None,
                    "reason_for_stopping": None
                })
                utterance_rows.append({
                    "speaker": "Child",
                    "utterance": child_response,
                    "stopper_decision": decision,
                    "reason_for_stopping": explanation
                })

                if decision != "continue":
                    logging.info(f"Evaluator decided to stop the conversation: {explanation}")
                    break

            file.write("\nConversation ended.\n\n")

        return utterance_rows



    # def execute_variants(self, query, variants, child_template=None, parent_template=None, output_file="variant_results.txt"):
    #     results = {}
    #     with open(output_file, "w") as file:  # Open in write mode to clear previous content
    #         for variant_name, config in variants.items():
                
    #             # Reset refusal count before starting a new variant
    #             self.refusals = 0  
    #             # self.disengagement_probability = 50  # Reset disengagement probability as well
            
    #             interaction = self.run_variant(query, config, child_template, parent_template)
    #             results[variant_name] = interaction

    #     return results

    

    def execute_variants(self, query, variants, child_template=None, parent_template=None, 
                     output_dialogue_csv="dialogue_dataset.csv", 
                     output_attributes_csv="attributes_dataset.csv", iterations=1):
        """
        For each iteration and each variant:
        - Re-randomizes the parent and child attributes and the engagement score.
        - Runs a conversation simulation.
        - Collects dialogue data (one row per utterance) with an ID of the format: T#-I#-V#-E#.
        - Also collects attributes information per variant (one row per variant per iteration).
        Finally, saves two CSV files: one for dialogue and one for attributes.
        """
        dialogue_rows = []
        attribute_rows = []
        topic_id = 1  # Assuming one topic; change as needed.

        # Enumerate through iterations
        for iter_index in range(iterations):
            # Re-randomize attributes and engagement score at the beginning of each iteration
            user_profiles = randomize_attributes(parent_template, True)
            child_profiles = randomize_attributes(child_template, False)
            # Here you can choose how to randomize the engagement score.
            # For example, using random.choice from your definitions list or generating a random float:
            self.engagement_score = random.choice([
                "Highly Engaged (1). This means that the child is very enthusiastic, actively participates in the conversation.",
                "Moderately Engaged (0.5). This means that the child shows a fair amount of interest, responds to questions, and contributes to the discussion but may not always reciprocate the energy.",
                "Neutral (0). This means that the child is neither particularly interested nor disengaged, responding when prompted but not initiating much on their own.",
                "Moderately Disengaged (-0.5). This means that the child shows reluctance to engage, giving short or minimal responses and displaying little interest in continuing the conversation.",
                "Highly Disengaged (-1). This means that the child is completely uninterested, avoids responding, and may actively try to end or leave the conversation."
            ])
            logging.info(f"Iteration {iter_index+1} - New engagement score: {self.engagement_score}")
            
            # For each variant in this iteration
            for variant_id, (variant_name, config) in enumerate(variants.items(), start=1):
                # If the variant configuration expects profiles, update them with the new random values
                if config.get("user_profiles") is not None:
                    config["user_profiles"] = user_profiles
                if config.get("child_profiles") is not None:
                    config["child_profiles"] = child_profiles

                # Reset any counters if necessary
                self.refusals = 0
                # Run the conversation variant; get list of utterance dictionaries.
                utterances = self.run_variant(query, config, child_template, parent_template)
                # Add an exchange counter for each utterance
                for ex_index, utterance_dict in enumerate(utterances, start=1):
                    # Construct an ID: T{topic_id}-I{iteration}-V{variant_id}-E{exchange_number}
                    utterance_id = f"T{topic_id}-I{iter_index+1}-V{variant_id}-E{ex_index}"
                    utterance_dict["ID"] = utterance_id
                    dialogue_rows.append(utterance_dict)

                # Prepare attributes row for this variant execution.
                attr_row = {
                    "ID": f"T{topic_id}-I{iter_index+1}-V{variant_id}",
                    "Variant": variant_name,
                    "Parent_Attributes": json.dumps(user_profiles) if config.get("user_profiles") is not None else "",
                    "Child_Attributes": json.dumps(child_profiles) if config.get("child_profiles") is not None else "",
                    "Guidelines": config.get("guidelines", ""),
                    "Reddit Context Included": config.get("context", False)
                }
                attribute_rows.append(attr_row)

        # Save dialogue rows to CSV
        dialogue_df = pd.DataFrame(dialogue_rows, columns=["ID", "speaker", "utterance", "stopper_decision", "reason_for_stopping"])
        dialogue_df.to_csv(output_dialogue_csv, index=False)
        logging.info(f"Dialogue dataset saved to {output_dialogue_csv}")

        # Save attributes rows to CSV
        attributes_df = pd.DataFrame(attribute_rows, columns=["ID", "Variant", "Parent_Attributes", "Child_Attributes", "Guidelines", "Reddit Context Included"])
        attributes_df.to_csv(output_attributes_csv, index=False)
        logging.info(f"Attributes dataset saved to {output_attributes_csv}")

        return dialogue_rows, attribute_rows


    



def main():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    index_file = 'reddit_index'
    index = None
    if os.path.exists(index_file):
        storage_context = StorageContext.from_defaults(persist_dir=index_file)
        index = load_index_from_storage(storage_context)

    # random_engagement_score = round(random.uniform(-1, 1), 1)
    

    definitions = ["Highly Engaged (1). This means that the child is very enthusiastic, actively participates in the conversation.",
                   "Moderately Engaged (0.5). This means that the child shows a fair amount of interest, responds to questions, and contributes to the discussion but may not always reciprocate the energy.",
                   "Neutral (0). This means that the child is neither particularly interested nor disengaged, responding when prompted but not initiating much on their own.",
                   "Moderately Disengaged (-0.5). This means that the child shows reluctance to engage, giving short or minimal responses and displaying little interest in continuing the conversation.",
                   "Highly Disengaged (-1). This means that the child is completely uninterested, avoids responding, and may actively try to end or leave the conversation."]
    
    random_engagement_score = random.choice(definitions)
    # random_engagement_score = "Highly Disengaged (-1). This means that the child is completely uninterested, avoids responding, and may actively try to end or leave the conversation."

    # random_engagement_score = -0.8

    generator = VariantGenerator(api_key, base_url, index=index, engagement_score=random_engagement_score)
    logging.info(f"Engagement Score set to {random_engagement_score}% for all variants.")


    # query = generator.generate_query()
    query = "Body Image & Self-Esteem"
    if not query:
        logging.error("No query retrieved from Excel. Exiting.")
        return

    """
    Guidelines: This is what would go into the prompt for the variants that have it enabled:

    Follow these guidelines while framing your response.

    Share information that matches your child’s age and understanding level.
    Keep responses short and simple for younger kids; older kids may need more depth.
    Focus on one topic at a time since too many topics can be overwhelming.
    Use concrete examples for younger kids and more abstract discussions for older ones.

    Ask open-ended questions to encourage discussion.
    Avoid jumping to conclusions about why your child is asking something. 
    Keep answers brief and clear, explaining any new words they may not know.
    Check their understanding by asking follow up questions.
    
    """
    variants = {
        "Variant 1 - Basic": {"context": False, "guidelines": None, "user_profiles": None, "child_profiles": None},
        "Variant 2 - Query + Reddit Context": {"context": True, "guidelines": None, "user_profiles": None, "child_profiles": None},
        "Variant 3 - Query + Guidelines": {
            "context": False,
            "guidelines": """Follow these guidelines while framing your response.
                            Don’t jump to conclusions about why they’re asking what they’re asking. ((For example, you can say: “Can you tell me what you already know about that?” or “What have you heard about that?” ))
                            Keep your answers short and simple, and explain new words that your kid might not have heard before.
                            Check their understanding. ((After answering a question for example, you can ask, “Does that answer your question?” or “What do you think about that?))""",
            "user_profiles": None,
            "child_profiles": None
        },
        "Variant 4 - Query + User Profiles": {
            "context": False,
            "guidelines": None,
            "user_profiles": True,
            "child_profiles": True
        },
        "Variant 5 - Query + Context + Guidelines": {
            "context": True,
            "guidelines": """Follow these guidelines while framing your response.
                            Don’t jump to conclusions about why they’re asking what they’re asking. ((For example, you can say: “Can you tell me what you already know about that?” or “What have you heard about that?” ))
                            Keep your answers short and simple, and explain new words that your kid might not have heard before.
                            Check their understanding. ((After answering a question for example, you can ask, “Does that answer your question?” or “What do you think about that?))""",
            "user_profiles": None,
            "child_profiles": None
        },
        "Variant 6 - Query + Context + User Profiles": {
            "context": True,
            "guidelines": None,
            "user_profiles": True,
            "child_profiles": True
        },
        "Variant 7 - Query + Guidelines + User Profiles": {
            "context": False,
            "guidelines": """Follow these guidelines while framing your response.
                            Don’t jump to conclusions about why they’re asking what they’re asking. ((For example, you can say: “Can you tell me what you already know about that?” or “What have you heard about that?” ))
                            Keep your answers short and simple, and explain new words that your kid might not have heard before.
                            Check their understanding. ((After answering a question for example, you can ask, “Does that answer your question?” or “What do you think about that?))""",
            "user_profiles": True,
            "child_profiles": True
        },
        "Variant 8 - Query + Context + Guidelines + User Profiles": {
            "context": True,
            "guidelines": """Follow these guidelines while framing your response.
                            Don’t jump to conclusions about why they’re asking what they’re asking. ((For example, you can say: “Can you tell me what you already know about that?” or “What have you heard about that?” ))
                            Keep your answers short and simple, and explain new words that your kid might not have heard before.
                            Check their understanding. ((After answering a question for example, you can ask, “Does that answer your question?” or “What do you think about that?))""",
            "user_profiles": True,
            "child_profiles": True
        }
    }

    generator.execute_variants(query, variants, child_template, parent_template, 
                                 output_dialogue_csv="dialogue_dataset.csv", 
                                 output_attributes_csv="attributes_dataset.csv", iterations=1)

    logging.info("Variant generation completed. Results saved to CSV files.")

if __name__ == "__main__":
    main()
