# **Variant Generator for Parent-Child Conversations**

## **Overview**
The Variant Generator is a Python-based program designed to simulate realistic parent-child conversations on sensitive topics, particularly sexual health and relationships. It utilizes GPT-4o-based agents to generate conversations, evaluate interactions, and analyze multiple conversation variants. The program randomizes conversational dynamics by varying attributes such as parent confidence, child engagement, gender, and conversation structure**.

This tool is designed for research purposes to help analyze how different conversational strategies impact child engagement, learning, and information retention.

---

## **Key Features**
* **Generates multiple conversational variants** based on different conditions (e.g., context availability, presence of guidelines, and user profiles).  
* **Uses AI agents** to simulate parent and child dialogue with realistic responses.  
* **Implements stopping criteria** to determine when a conversation should end based on disengagement, goal completion, or mutual agreement.  
* **Supports randomized user attributes** (parent’s confidence, comfort level, dialogue openness, child’s age, gender, and closeness level).  
* **Retrieves real-world Reddit context** for relevant topic discussions (if enabled).  
* **Evaluates engagement levels** of child responses using predefined engagement scoring.  
* **Exports conversation data** into structured CSV files for further analysis.

---

## **How It Works**
The **Variant Generator** follows these steps:

1. **Loads pre-defined conversation topics** from an Excel file.
2. **Randomly generates parent and child attributes**, such as:
   - **Parent confidence, comfort, and openness levels.**
   - **Child’s age, gender identity, and closeness to parent.**
3. **Retrieves external context** from Reddit (if applicable).
4. **Runs a turn-based AI-driven conversation** between a parent and child.
5. **Evaluates the conversation** after each turn using a third AI agent (evaluator):
   - Determines **whether to continue or stop the conversation** based on rules.
   - Checks for **child disengagement, goal completion, or mutual agreement.**
6. **Saves conversation logs** with structured identifiers (e.g., `T1-I3-V4-E7`).
7. **Generates CSV files** for further research and analysis.

---

## **Conversation Variants**
The program supports **8 distinct conversation variants** to evaluate different conditions:

| Variant | Context Retrieval | Guidelines | User Profiles |
|---------|------------------|------------|--------------|
| **Variant 1** | ❌ | ❌ | ❌ |
| **Variant 2** | ✅ | ❌ | ❌ |
| **Variant 3** | ❌ | ✅ | ❌ |
| **Variant 4** | ❌ | ❌ | ✅ |
| **Variant 5** | ✅ | ✅ | ❌ |
| **Variant 6** | ✅ | ❌ | ✅ |
| **Variant 7** | ❌ | ✅ | ✅ |
| **Variant 8** | ✅ | ✅ | ✅ |

**Explanation:**
- **Context Retrieval** → Uses external knowledge from Reddit posts.
- **Guidelines** → Provides structured guidance for conversations.
- **User Profiles** → Uses specific parent and child attributes.

---

## **File Outputs**
The program generates **two primary CSV files**:

### 1️⃣ **`dialogue_dataset.csv`**  
Contains **detailed conversation data**, including:
- **ID**: Unique identifier (`T#-I#-V#-E#` format).
- **Speaker**: "Parent" or "Child".
- **Utterance**: The actual spoken text.
- **Stopping Criteria**: Whether the conversation stopped.
- **Reason for Stopping**: Explanation for stopping.
---

### 2️⃣ **`attributes_dataset.csv`**  
Contains **variant-specific metadata**, including:
- **Parent and child attributes** (confidence, comfort, gender, etc.).
- **Whether Reddit context was used.**
- **Guidelines applied for the conversation.**
---

## **Stopping Criteria**
The evaluator AI determines **when to stop** the conversation based on these **rules**:

1. **Child Disengagement** → If the child disengages **at least three times**, the conversation ends.
2. **Goal Met** → If the conversation meets its educational goal, it stops.
3. **Mutual Agreement** → If both the parent and child **agree to stop**.

---

## **Additional Features**
* **Engagement Scoring:**  
   - The child's engagement level is randomly assigned from five categories:
     - **Highly Engaged (1)** → Actively participates.
     - **Moderately Engaged (0.5)** → Responds but isn’t very involved.
     - **Neutral (0)** → Does not show strong interest or disinterest.
     - **Moderately Disengaged (-0.5)** → Minimal responses, little interest.
     - **Highly Disengaged (-1)** → Actively avoids or tries to end the conversation.

* **AI-Powered Conversational Flow:**  
   - The **parent's responses are realistic and natural**.
   - The **child's responses reflect Generation Alpha speech styles**.
   - Conversations stay **concise, on-topic, and engaging**.

* **Real-World Context Integration:**  
   - **Reddit-based contextual retrieval** helps provide **real-world discussion points**.

* **Multiple Iterations Per Topic:**  
   - **Runs the same conversation multiple times** with different attributes.

---

## **Installation & Setup**
### **1 Prerequisites**
- Python 3.8+
- OpenAI API Key (`OPENAI_API_KEY`)
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

### **2 Running the Script**
```bash
python ./variant_generator.py
```

---

## **How to Customize**
* **Modify the Number of Iterations**  
In `main()`, update:
```python
iterations = 50  # Change this to any number
```

* **Change the Default Topics and Topic ID**  
Modify `EXCEL_FILE_PATH` in:
```python
EXCEL_FILE_PATH = "filtered_topics.xlsx"
```
Or manually set a topic in:
```python
query = "Teen Healthy Relationships"
```
---

## **Authors**
- **Aryan Santhosh Kumar and Benyamin Tabarsi**  

---
