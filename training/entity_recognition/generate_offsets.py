import spacy
import json
import os

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Load sentences from JSON file
def load_sentences(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Path to the sentences JSON file
sentences_file_path = os.path.join(os.path.dirname(__file__), "./tmp/sentences.json")
sentences_data = load_sentences(sentences_file_path)

# Convert loaded data to the required format
sentences = [
    (item["text"], {"entities": [(ent["text"], ent["label"]) for ent in item["entities"]]})
    for item in sentences_data
]

def generate_offsets(sentences):
    training_data = []
    for text, annotations in sentences:
        doc = nlp(text)
        entities = []
        for ent_text, label in annotations["entities"]:
            for token in doc:
                if token.text.lower() == ent_text.lower():
                    start = token.idx
                    end = token.idx + len(token.text)
                    entities.append((start, end, label))
        training_data.append({"text": text, "entities": entities})
    return training_data

# Generate and save the training data
training_data = generate_offsets(sentences)
output_file = "./entity_recognition/tmp/training_data.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Training data generated and saved to {output_file}")
