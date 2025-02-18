# **ParenTeenTalk: A Framework and Benchmark Dataset Leveraging Crowd-Sourced Topics And Research-Based Guidelines To Generate And Evaluate Parent-Teen Health Conversations**
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/benneigh/ParenTeenTalk)

[![GitHub issues](https://img.shields.io/github/issues/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/issues)
[![GitHub forks](https://img.shields.io/github/forks/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/network)
[![GitHub stars](https://img.shields.io/github/stars/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/stargazers)
[![GitHub license](https://img.shields.io/github/license/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/blob/main/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/benneigh/ParenTeenTalk)](https://github.com/benneigh/ParenTeenTalk/graphs/contributors)
[![Code Coverage](https://img.shields.io/codecov/c/gh/benneigh/ParenTeenTalk)](https://codecov.io/gh/benneigh/ParenTeenTalk)
[![Build Status](https://img.shields.io/github/workflow/status/benneigh/ParenTeenTalk/CI)](https://github.com/benneigh/ParenTeenTalk/actions)
[![Dependabot Status](https://img.shields.io/badge/dependabot-active-brightgreen)](https://github.com/benneigh/ParenTeenTalk/pulls?q=is%3Apr+is%3Aopen+label%3A%22dependencies%22)
## **Overview**
ParenTeenTalk is a multi-agent AI framework that generates realistic parent-teen conversations about sexual health. It combines expert-reviewed guidelines, retrieval-augmented generation (RAG) from Reddit, and structured evaluation metrics to ensure conversations are natural, informative, and developmentally appropriate. This dataset includes 8,000 conversations across 20 sexual health topics, with eight dataset variants to study how different conversational factors (e.g., parental confidence, child engagement) impact discussion dynamics. ParenTeenTalk dataset and data generation pipeline offer a scalable, structured, and practical resource for AI healthcare, and human-centered research in sensitive topics.

**Disclaimer:** This dataset is for **research and educational purposes** and should not be regarded as a substitute for professional medical or psychological advice.
---

## **Key Features**
- **Generates Multiple Conversational Variants**  
  Varies auxiliary components (e.g., confidence level, child closeness, engagement) to create a rich set of simulated dialogues.
- **Multi-Agent Setup**  
  Employs **Parent**, **Child**, and **Moderator** AI agents for realistic turn-based conversations.
- **Retrieval-Augmented Generation (RAG)**  
  Integrates **real-world forum data** (Reddit) to ground conversations in authentic concerns.
- **Evidence-Based Guidelines**  
  Uses **research-driven** tips (e.g., Planned Parenthood’s recommendations) to shape conversation flow.
- **Stopping Criteria**  
  Terminates conversations naturally (child disengagement, mutual agreement, goal completion, or turn limit).
- **Structured Evaluation**  
  Compares dataset variants on **conversation structure, semantic quality, and communication effectiveness** (e.g., developmental appropriateness, content safety, and adherence to guidelines).
- **Scalable for Future Research**  
  Easily adapted for **other sensitive topics** or **health contexts** beyond sexual health.
---

## **How It Works**
The **Variant Generator** follows these steps:

1. **Topic and Data Loading**  
   - Reads a set of **20 sexual health topics** from an external file (e.g., Excel/CSV).  
   - Optionally retrieves **Reddit-based contextual data** when context retrieval is enabled.
2. **Attribute Randomization**  
   - Generates **Parent** attributes: confidence, comfort, openness.  
   - Generates **Child** attributes: age (10–13), gender, engagement level, closeness to parent.
3. **External Context Retrieval** from Reddit (if applicable).
4. **Conversation Simulation**  
   - **Parent** and **Child** agents exchange turns, maintaining **natural** and **on-topic** dialogue.  
5. **Moderator Evaluation** 
   - After each turn using a third AI agent (moderator), determines **whether to continue or stop the conversation** based on rules.
   - It checks for **child disengagement, goal completion, mutual agreement or hitting the turn limit.**
6. **Logging**  
   - Saves **conversation logs** with structured identifiers (`T1-I3-V4-E7`)—T for topic, I for iteration, V for variant and E for exchange.
   - Logs **stopping criteria** (e.g., goal completion, disengagement).  
   - Stores **metadata** (variant settings, user attributes) in structured CSV files.
7. **Generates CSV files** for further research and analysis.

---

## **Conversation Variants**
ParenTeenTalk supports **8 distinct variants** to explore how different configurations affect conversation quality:

| Variant | Context Retrieval | Guidelines | User Profiles |
|---------|-------------------|------------|--------------|
| **1**   | ❌                | ❌         | ❌           |
| **2**   | ✅                | ❌         | ❌           |
| **3**   | ❌                | ✅         | ❌           |
| **4**   | ❌                | ❌         | ✅           |
| **5**   | ✅                | ✅         | ❌           |
| **6**   | ✅                | ❌         | ✅           |
| **7**   | ❌                | ✅         | ✅           |
| **8**   | ✅                | ✅         | ✅           |

- **Context Retrieval:** Uses Reddit data to inform dialogue.  
- **Guidelines:** Incorporates expert-driven tips (e.g., Planned Parenthood).  
- **User Profiles:** Employs randomized parent/child attributes (confidence, comfort, age, etc.).

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
4. **Turn Limit**  → After **10 total turns**, the conversation automatically ends.
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
- [Benyamin Tabarsi](https://benyamintabarsi.com)
- [Aryan Santhosh Kumar](https://www.linkedin.com/in/aryan-sk)
- [Dr. Dongkuan (DK) Xu](https://dongkuanx27.github.io/)
- [Dr. Laura Widman](https://www.drlaurawidman.com/)
- [Dr. Tiffany Barnes](https://eliza.csc.ncsu.edu/)
---
