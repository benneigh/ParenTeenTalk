import json

def transform_entry(entry):
    # Handles the transformation of a single entry to the desired format
    return {
        "title": entry.get("title", ""),
        "content": entry.get("selftext") or entry.get("content", ""),
        "most_upvoted_comment": (
            entry.get("most_upvoted_comment") or
            (entry.get("top_comment", [{}])[0].get("comment_body") if entry.get("top_comment") else "")
        ),
        "url": entry.get("url", "")
    }

def merge_json_files(file1, file2, output_file):
    # Load the first JSON file
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    # Load the second JSON file
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    # Transform both sets of data to the desired format
    transformed_data1 = [transform_entry(entry) for entry in data1]
    transformed_data2 = [transform_entry(entry) for entry in data2]
    
    # Combine the two lists
    combined_data = transformed_data1 + transformed_data2
    
    # Save to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined JSON file created: {output_file}")

# File paths for input and output
file1 = 'parenting_1.json'
file2 = 'parenting_2.json'
output_file = 'r_parenting.json' 

# Run the merge function
merge_json_files(file1, file2, output_file)