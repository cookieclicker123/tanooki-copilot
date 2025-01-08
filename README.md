# @tanooki - AI copilot for Tanooki

## Vision

```bash
“The AI sidekick that knows your production inside out and suggests what you need next.”
    • Always On, Always Ready

24/7 expert guidance without the wait.
    • Perfect Memory

Remembers every decision, timeline, and detail so you don’t have to.
    • Smart Suggestions

Proactively recommends follow-up queries and next steps.
    • A Tireless Tutor

Get to the answers you want at all times, no judgement or fatigue.
    • Voice-Enabled Interaction

Speak or type—it’s your choice for instant insights.
```

## What i need to build

```bash
Specialised Intent and Data Extraction ML for Tanooki use case

Media Search Engine Integration - For production queries convert the Intent and Data requirements into a Search Request and handle the response 

Tanooki Knowledge Base - Use to get information back to the user for requests on how to do stuff in Tanooki 

Fine tuned LLM for Industry savvy AI - Understands the workflows, paradigms and vernacular of the industry
``` 

## Initial Setup

```bash
git clone git@github.com:cookieclicker123/tanooki-copilot.git
cd tanooki-copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 - building Agent Intent Recognition

### Prerequisites

Before training the intent recognition model, ensure you have the necessary spaCy model and lookup data installed. Run the following commands:

```bash
python -m spacy download en_core_web_lg

pip install spacy-lookups-data
```

### Train the Intent Recognition Model

```bash
python intent_recognition/train_intent.py
```

### Test the Intent Recognition Model

```bash
pytest tests/test_intent.py 

#View the logs in tmp/intent_logs/ for predictions on each query
```

