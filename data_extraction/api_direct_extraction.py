import praw
import json
import time
from datetime import datetime, timedelta

# Reddit API credentials
CLIENT_ID = 'NR8Ea4hGW6YwQbX2KISYog'
CLIENT_SECRET = 'mNGk0Qwn2Gleb7DxSY7GMVTISLNMdw'
USER_AGENT = 'aryxsk'

# Set up Reddit API client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Define the subreddit, total number of posts to retrieve, and keywords
subreddit_name = 'parenting'
total_posts = 10000
posts_per_batch = 10000  # Number of posts per batch
pause_duration = 10  # Pause duration in seconds (1 hour)
keywords = ["sexual health", "sex", "birth control", "teen sex", "STDs", "safe sex", "puberty", "consent", "sexual consent", "first-time sex", "teen pregnancy", "peer pressure", "HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Herpes", "HPV", "IUD", "condom", "Plan B", "abstinence", "LGBTQ", "gender identity", "pornography", "sexting"]

# Initialize variables
processed_ids = set()
posts = []
processed_posts = 0
days_per_batch = 10  # Narrowing to 1 day per batch
current_end_date = datetime.now()

while processed_posts < total_posts:
    try:
        for keyword in keywords:
            current_start_date = current_end_date - timedelta(days=days_per_batch)
            start_epoch = int(current_start_date.timestamp())
            end_epoch = int(current_end_date.timestamp())

            # Construct the search query with after and before
            submissions = reddit.subreddit(subreddit_name).search(
                query=f"{keyword}",
                sort='new',
                limit=1000,
                params={"before": end_epoch, "after": start_epoch}
            )

            batch_empty = True
            for submission in submissions:
                batch_empty = False
                if submission.id not in processed_ids:
                    submission.comment_sort = 'top'
                    submission.comments.replace_more(limit=0)

                    # Find the most upvoted non-mod comment
                    top_comment_text = "No suitable comment found"
                    try:
                        for comment in submission.comments:
                            # Skip the comment if it's from a mod (if the flag exists)
                            if hasattr(comment, 'distinguished') and comment.distinguished == 'moderator':
                                continue  # Skip moderator comment
                            
                            # If no 'distinguished' flag or it's not a mod, store the comment
                            if comment.author is not None:  # Check if the author exists
                                top_comment_text = comment.body
                                break  # Stop after finding the first suitable comment
                    except AttributeError as e:
                        print(f"Error processing comment in post ID: {submission.id}: {e}")
                        # Skip the rest of the comments and continue processing other posts
                        continue

                    post_data = {
                        'title': submission.title.strip(),
                        'content': submission.selftext.strip(),
                        'most_upvoted_comment': top_comment_text.strip(),
                        'url': submission.url  # Added URL of the submission
                    }
                    posts.append(post_data)
                    processed_ids.add(submission.id)
                    processed_posts += 1

                    print(processed_posts)
                    print("\n")

                # Break the loop if we reach the post limit for this batch
                if processed_posts % posts_per_batch == 0:
                    break

            if batch_empty:
                print(f"No more posts available for '{keyword}' in this timeframe.")
            else:
                # Update the end date for the next batch
                current_end_date = current_start_date

            # Save progress incrementally to avoid data loss
            output_file = f"{subreddit_name}_1.json"
            with open(output_file, 'w') as f:
                json.dump(posts, f, indent=4)

            print(f"Processed {processed_posts} unique posts, saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(60)  # Pause and retry if an error occurs

print(f"Completed processing {processed_posts} unique posts.")