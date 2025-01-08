import spacy
from typing import Dict
from .data_types import ExtractedEntities, EntityExtractionFn

def entity_extraction_factory(model_path: str = "./tmp/models/ner_model") -> EntityExtractionFn:
    """Factory function to create an entity extraction function with a preloaded model."""
    nlp = spacy.load(model_path)

    def extract_entities_from_query(query: str, available_entities: Dict) -> ExtractedEntities:
        doc = nlp(query)
        contributors = []
        locations = []
        clip_types = []

        # Extract entities using spaCy's NER
        for ent in doc.ents:
            if ent.label_ == "CONTRIBUTOR":
                for contrib_id, contrib_name in available_entities["contributors"].items():
                    if ent.text.lower() == contrib_name.lower():
                        contributors.append(contrib_id)
            elif ent.label_ == "LOCATION":
                for loc_id, loc_name in available_entities["locations"].items():
                    if ent.text.lower() == loc_name.lower():
                        locations.append(loc_id)
            elif ent.label_ == "CLIP_TYPE":
                # If a generic term like "clips" is used, include all available clip types
                if ent.text.lower() == "clips":
                    clip_types.extend(available_entities["clip_types"])
                else:
                    clip_types.append(ent.text.lower())

        # Return extracted entities
        return ExtractedEntities(
            entities={
                "contributors": contributors,
                "locations": locations,
                "clip_types": clip_types
            },
        )

    return extract_entities_from_query

# Test function
if __name__ == "__main__":
    available_entities = {
        "contributors": {
            "john_id": "John",
            "david_id": "David",
            "sarah_id": "Sarah"
        },
        "locations": {
            "beach_id": "Beach",
            "kitchen_id": "Kitchen"
        },
        "clip_types": ["rush", "review"]
    }

    # Create the entity extraction function
    entity_fn = entity_extraction_factory()

    query = "Find me all the clips where John is at the Beach."
    extracted_entities = entity_fn(query, available_entities)
    print(extracted_entities)
