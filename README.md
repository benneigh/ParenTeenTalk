**November 3:**

- Initial push to repo: consists of python script to extract reddit data, RAG agent, vector indexes based on previously generated r/teenagers data
- Obtained two examples of data from r/parenting through BigQuery and personal script.
- Formatted that r/parenting data. Ended up with 1205 posts!
- Total number of posts for data context: _1205 (parenting) + 4468 (teenagers and teenagers_new) = **5673** posts_

Total count of all posts is now 5673 posts. Started data pre-processing. Created numerous scripts:
1.Quality Filter: Sentiment Analysis of whether or not a post is a "Rant"
2.Question Detection: Identifying question marks and common question words
3.Scenario Detection: Pronoun + Verb analysis to detect scenario posts.
4.Advice Extraction: Keyword recognition.
5.Cluster Analysis
6.Topic Analysis
7.Individual Word Count
8.LLM Analysis
