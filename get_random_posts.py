import json
import random

# Function to extract sentences from the JSON file
def extract_sentences_from_json(file_path):
    sentences = []

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i % 1000 == 0:
                print(str(i/5024000))
            m = random.randint(0, 9999)
            if (m <= 220):
                try:
                    tweet = json.loads(line.strip())
                    sentences.append(tweet)
                except json.JSONDecodeError:
                    continue
    return sentences

# Function to randomly select sentences and write to a new file
def write_random_sentences_to_file(sentences, output_file, num_sentences):
    selected_sentences = random.sample(sentences, min(num_sentences, len(sentences)))
    with open(output_file, 'w') as file:
        for sentence in selected_sentences:
            file.write(json.dumps(sentence) + '\n')

# Path to the input JSON file
input_file = 'datasets/twitter.jsonl'
# Path to the output file
output_file = 'datasets/full_random_dataset.txt'
# Number of random sentences to select
num_sentences = 100000

# Extract sentences from the JSON file
sentences = extract_sentences_from_json(input_file)

print("finished with extraction")
# Write random sentences to the output file
write_random_sentences_to_file(sentences, output_file, num_sentences)
