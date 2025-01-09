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

### Swap in queries such as the following to test the tv_model against base llama3.2:3B.

 - "In a distributed remote post-production workflow with multiple editors and colorists, how do you handle timecode drift in multi-camera footage while maintaining accurate sync across HDR and SDR deliverables? What tools and pipeline adjustments would you recommend to minimize issues?"

 - "In a multi-location shoot where teams are using both RED and ARRI cameras, how do you maintain timecode sync across devices, especially when there is no master clock available? What post-production tools can align metadata for such footage, and what are their limitations?"

 - "When downmapping Dolby Vision HDR to SDR using DaVinci Resolve, what are the optimal trim pass settings to ensure consistent brightness and contrast without losing highlight details? Include a discussion of MaxCLL and MaxFALL considerations."

 - "What are the key challenges when working with ACES color management in a VFX-heavy pipeline, and how do you ensure consistency across HDR and SDR deliverables when integrating EXR sequences from multiple vendors?"

 - "For an OTT platform that prioritizes AV1 for streaming but needs backward compatibility with H.264, how do you configure a transcoding pipeline to maximize compression efficiency while ensuring playback stability across devices?"

 - "In a collaborative workflow using cloud-based MAM systems like Frame.io, how do you automate proxy generation, metadata tagging, and version control to streamline asset handoffs between editors and colorists?""

 - "What are the benefits of using multicam editing workflows for interviews, and how do you handle syncing audio when timecode is unavailable?"

 - "What are the key steps in a standard post-production workflow for a 30-second commercial, and how can you ensure all stakeholders approve final edits efficiently?"

 - "What are the differences between ProRes 422 HQ and ProRes 4444, and when would you choose one over the other for a short film project?"

 - "Why is it important to use calibrated monitors for color grading, and how can you ensure your grading setup complies with Rec.709 standards for broadcast?"

 - "In a tight deadline situation for delivering episodic content, how do you prioritize tasks between editing, color grading, and sound design to meet the delivery schedule?"

 