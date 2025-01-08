import spacy
import json

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Sample sentences with entity annotations by text
sentences = [
    ("Find me all the clips where John is at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Show all the video segments with John at the Beach.", {"entities": [("segments", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Can you find the clips where John is located at the Beach?", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Pull up all clips featuring John at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("List the clips where John appears at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("I need all the clips that have John at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Fetch the clips of John shot at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Retrieve all clips where John is seen at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Get me the video clips that show John at the Beach.", {"entities": [("video clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
    ("Please find clips of John at the Beach.", {"entities": [("clips", "CLIP_TYPE"), ("John", "CONTRIBUTOR"), ("Beach", "LOCATION")]}),
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
