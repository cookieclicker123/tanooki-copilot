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

## Phase 2 - building Entity Recognition Model

### Generate Training Data

```bash
python entity_recognition/generate_offsets.py
```

### Train the Entity Recognition Model

```bash
python entity_recognition/train_ner.py
```

### Test the Entity Recognition Model

```bash
pytest tests/test_entities.py

#View the logs in tmp/entity_logs/ for predictions on each query
```

## Phase 3 - building the TV Expert Model

 - upload the ipynb file inside tv_expertto colab and run it, ensure you are using sufficient GPU resources
Ideally an A100
 - Play around with the parameters to get the best results
 - Make sure to download all the model files in tv_post_3b. Ideally move them to drive first to prevent running out of space in colab
 - Make sure you give drive sufficient time to load the files, leave them for 30 mins to fully load.
 - Once you have them locally on disk, move them to a folder called tv_model in tmp.
 - Install ollama if you havent
 - then in terminal run ollama list, there should be nothing if youve just installed
 - Then run the following commands:

```bash
ollama list

ollama create ./tmp/tv_model

ollama list

ollama run tv_model 
```

Ask the model any question in realtion to TV Post Production, general or specific.
It's already very good, despite being ultra lightweight, local , private cloud and free.
No external api's or third party liscenes are needed to create domain expertise.

### Run the formal Test

```bash
python tests/test_prompt.py --query "What are the fundamental concepts i should know about timecode and what are practical workflows and software i should know about" --provider ollama --model tv_model:latest
```