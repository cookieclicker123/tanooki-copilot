import spacy
from spacy.training.example import Example
import json
import os

# Load the base model
base_model = "en_core_web_lg"
nlp = spacy.load(base_model)

# Add a new NER pipe if not already present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Load the dynamically generated training data
with open("./entity_recognition/tmp/training_data.json", "r") as f:
    TRAINING_DATA = json.load(f)

# Add entity labels to the NER component
labels = ["CONTRIBUTOR", "LOCATION", "CLIP_TYPE"]
for label in labels:
    ner.add_label(label)

# Training the model
def train_model(nlp, data, output_dir, n_iter=20):
    # Create training examples
    examples = []
    for item in data:
        text = item["text"]
        entities = [(ent[0], ent[1], ent[2]) for ent in item["entities"]]
        examples.append(Example.from_dict(nlp.make_doc(text), {"entities": entities}))

    # Disable other pipes during training
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.resume_training()
        for i in range(n_iter):
            losses = {}
            for example in examples:
                nlp.update([example], sgd=optimizer, losses=losses)
            print(f"Iteration {i + 1}/{n_iter}, Losses: {losses}")

    # Save the trained model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

# Train and save the model
output_dir = "./tmp/models/ner_model"
train_model(nlp, TRAINING_DATA, output_dir)
