import csv
import re

def is_related_to_evolution(text):
    # Define a broader set of keywords related to evolution, excluding terms like "revolution"
    evolution_keywords = [
        "natural selection", "Darwin", "evolve", "species", "adaptation", "mutation",
        "genetics", "survival of the fittest", "inheritance", "variation", "speciation",
        "evolutionary", "fitness", "phylogeny", "common ancestor", "fossil record"
    ]
    exclusion_keywords = ["revolution", "revolutionary"]
    
    # Use regex to match keywords case-insensitively
    if any(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE) for keyword in evolution_keywords):
        # Check for exclusion keywords
        if not any(re.search(r'\b' + re.escape(exclusion_keyword) + r'\b', text, re.IGNORECASE) for exclusion_keyword in exclusion_keywords):
            return True
    return False

def filter_evolution_posts(input_file, output_file):
    with open(input_file, 'r', encoding='ISO-8859-1') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        total_lines = 0
        matched_lines = 0
        for row in reader:
            total_lines += 1
            if len(row) < 6:
                print(f"Line {total_lines} does not have enough columns")
                continue
            tweet_text = row[5]
            # Clean the text
            tweet_text = tweet_text.strip()
            if is_related_to_evolution(tweet_text):
                matched_lines += 1
                outfile.write(tweet_text + "\n")
        print(f"Total lines processed: {total_lines}")
        print(f"Total lines matched: {matched_lines}")

# Replace 'data.csv' with the path to your CSV file
input_file = 'datasets/data.csv'
# Output file where the filtered posts will be saved
output_file = 'datasets/evolution2.txt'

filter_evolution_posts(input_file, output_file)





# import json
# import re

# def is_related_to_evolution(text):
#     # Define a broader set of keywords related to evolution
#     evolution_keywords = [
#         "natural selection", "Darwin", "evolve", "species", "adaptation", "mutation",
#         "genetics", "survival of the fittest", "inheritance", "variation", "speciation",
#         "evolutionary", "fitness", "phylogeny", "common ancestor", "fossil record"
#     ]
#     # Use regex to match keywords case-insensitively
#     return any(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE) for keyword in evolution_keywords)

# def filter_evolution_posts(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         total_lines = 0
#         matched_lines = 0
#         for line in infile:
#             total_lines += 1
#             tweet = json.loads(line)
#             tweet_text = tweet.get("content", "")
#             if is_related_to_evolution(tweet_text):
#                 matched_lines += 1
#                 outfile.write(str(tweet) + "\n")
#         print(f"Total lines processed: {total_lines}")
#         print(f"Total lines matched: {matched_lines}")

# # Replace 'twitter.jsonl' with the path to your JSONL file
# input_file = 'datasets/twitter.jsonl'
# # Output file where the filtered posts will be saved
# output_file = 'datasets/evolution2.txt'

# filter_evolution_posts(input_file, output_file)