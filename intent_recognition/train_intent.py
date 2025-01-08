import spacy
from spacy.training.example import Example
import random
from tqdm import tqdm
import torch
import os

# Create directories if they don't exist
MODEL_PATH = "./tmp/models/agent_model"
os.makedirs("./tmp/models", exist_ok=True)

# Check for MPS availability
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("\nUsing MPS (Metal Performance Shaders) acceleration")
        return torch.device("mps")
    else:
        print("\nMPS not available, using CPU")
        return torch.device("cpu")

# Load larger model
nlp = spacy.load("en_core_web_lg")

# Set device for spaCy/PyTorch
device = get_device()
if device.type == "mps":
    spacy.require_gpu()
    # No need to explicitly set MPS device, it's handled by the device creation

# Training data with 5 variations for each of our 12 queries
TRAIN_DATA = [
    # Production Agent queries - Footage Query
    ("How much footage did we shoot for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What's the total amount of footage for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the footage duration for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What's the length of all footage in project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me the total hours of footage for project A", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for Footage Query
    ("Can you give me the total footage count for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the complete footage duration for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Tell me the total amount of footage captured for project A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How much total footage is there for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Provide the total footage length for project A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the overall footage volume for project A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Production Agent queries - Speaking Time
    ("Who spoke the most in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Which person had the most speaking time in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me who talked the most in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What's the speaking duration breakdown for production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me the top speaker from production A", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for Speaking Time
    ("Identify the person who spoke the most in production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Who had the highest speaking time in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Which individual talked the most in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Who logged the most speaking hours in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Identify the top speaker in production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Who dominated the speaking time in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # TV Post Production Agent queries - Camera Query
    ("How much of that was shot on the A7S?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What portion was filmed using the A7S camera?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you break down how much A7S footage we have?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What's the A7S footage duration?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about the A7S shots", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Invalid Query - Production B Access
    ("And how much for production B on the A7S?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("What's the footage count for production B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Can you show me production B's stats?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("I need to see production B's numbers", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Give me access to production B", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),

    # TV Post Production Agent queries - Timecode
    ("What's timecode used for?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain timecode to me?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How does timecode work?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about timecode synchronization", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What's the purpose of timecode?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for Timecode
    ("What is the function of timecode in video production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Explain how timecode is utilized in editing.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Describe the role of timecode in post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # Production Agent queries - Scene Search
    ("I recall a scene where michael was wearing a blue shirt at the beach holding a microphone", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you find the beach scene with Michael in blue?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Where's the clip of Michael at the beach with a mic?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find me the scene where Michael's wearing blue by the ocean", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Which scene has Michael in a blue shirt on the beach?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Production Agent queries - Scene Dialog
    ("What's the first thing he says in this scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me Michael's first line here?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What does Michael say at the start?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me the opening dialogue from this scene", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are Michael's first words?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Invalid Query - Crew List
    ("List each person in the production crew", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Show me the crew roster", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Who's on the production team?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Can I see the crew list?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),
    ("Give me the production staff details", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 1.0}}),

    # Production Agent queries - Swearing Clips
    ("Please retrieve me each clip where someone swears", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all instances of swearing", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me clips containing profanity", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Where are all the swear words in the footage?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List scenes with curse words", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Tanooki App Agent queries - Face Blur
    ("How can i retrieve all people logged to have their faces blurred from earlier, and perform the actual face blurring in the tanooki app?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Where do I find the face blur registry in Tanooki?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do I access the face blurring feature?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Show me how to blur faces in the Tanooki app", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Need to apply face blurring, how do I do it?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # Tanooki App Agent queries - Avid Bin
    ("Explain what an avid bin is and how i can export project A to a new bin in the tanooki app", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do I create a new bin in Tanooki?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you help me export to a Tanooki bin?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What's the process for bin exports in Tanooki?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about Avid bins and Tanooki exports", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 1.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # TV Post Production Agent queries - ADR
    ("How is ADR (Automated Dialogue Replacement) typically handled in TV post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What's the process for ADR in TV post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain ADR to me?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about ADR in TV post-production", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How is ADR typically handled in TV post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for ADR
    ("What are the steps involved in ADR for TV shows?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do TV productions manage ADR sessions?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Explain the ADR workflow in television post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # Production Agent Queries - follow up

    # And How many hours of footage is he in across the whole project? 5 differently asked examples
    ("And How many hours of footage is he in across the whole project?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many hours of footage is he in across the whole project?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many hours of footage is he in across the production?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many hours of footage is he in across the whole production?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How much does he appear in the production?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Production Agent queries - Summarize Production
    ("Summarize the details of Production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me a summary of Production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the key details of Production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you summarize Production A for me?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Provide a summary of Production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for absent queries
    ("shoe me the clips of david at the beach", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Display the clips of David at the beach.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show all beach clips featuring David.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find the footage of David at the beach.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("When was david at the beach?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What time was David at the beach?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me when David was at the beach?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Identify the time David was at the beach.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("show me the clips of sarah at the pool", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Display the pool clips with Sarah.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all footage of Sarah at the pool.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show the videos of Sarah by the pool.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("Which is the scene where john is talking to david?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find the scene where John converses with David.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the clip of John speaking to David.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Identify the scene with John and David talking.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

   

    # New variations for absent queries
    ("How many cameras do we flim on", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the number of cameras we use for filming?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many cameras are utilized in our productions?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the total number of cameras we film with?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("How many cameras do we film on in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the camera count for production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many cameras are used in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you list the cameras used in production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("How many city scenes do we have in production D?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the total number of city scenes in production D?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many scenes set in the city are there in production D?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the count of city scenes in production D?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("How much footage do we have recorded on the sony a7sii?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the total footage captured with the Sony A7SII?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How much video is recorded using the Sony A7SII?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide the footage amount from the Sony A7SII?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("Please return all the scenes where John AND david are present", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all scenes featuring both John and David.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the clips with John and David together.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Retrieve scenes where both John and David appear.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for absent queries
    ("Please return all the edits where sarah is", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all edits featuring Sarah.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the clips where Sarah appears.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Retrieve all scenes with Sarah.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("which shoot dates does Seb appear in?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("On which dates does Seb appear in the shoot?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Identify the shoot dates featuring Seb.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the dates Seb is present in the shoot.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("get me the original rushes for production A", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Retrieve the original rushes for production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide the original rushes for production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the original rushes from production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("How do I reduce rendering times in DaVinci Resolve?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What are the best practices to reduce rendering times in DaVinci Resolve?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How can I speed up rendering in DaVinci Resolve?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Tips for reducing rendering times in DaVinci Resolve?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("What's the best way to organize footage bins in Avid Media Composer?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How should I organize footage bins in Avid Media Composer?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Best practices for organizing footage bins in Avid Media Composer?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What is the optimal way to arrange footage bins in Avid Media Composer?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New examples for TV Post Production Agent queries
    ("What's the standard frame rate for broadcast TV in Europe?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What frame rate is typically used for European broadcast TV?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the standard frame rate for TV broadcasts in Europe?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What is the usual frame rate for broadcasting TV in Europe?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("How do I set up color grading for SDR and HDR outputs?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What are the steps to set up color grading for SDR and HDR?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How is color grading configured for both SDR and HDR outputs?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain the process of setting up color grading for SDR and HDR?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("What are the key differences between offline and online editing workflows?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do offline and online editing workflows differ?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What distinguishes offline editing from online editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain the differences between offline and online editing workflows?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("What's the standard loudness level for broadcast TV audio?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What is the typical loudness level for TV audio broadcasts?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the standard audio loudness for TV broadcasts?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What loudness level is used for broadcast TV audio?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("How do I set up proxy workflows for remote collaboration?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What are the steps to establish proxy workflows for remote editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How can I configure proxy workflows for working remotely?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain how to set up proxy workflows for remote collaboration?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    ("How much footage was shot for production A and B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # New variations for underrepresented queries
    ("How much footage was shot for production A and B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the total footage recorded for productions A and B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the combined footage for production A and B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How much video was captured across productions A and B?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("What's the first thing he says in this scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are his opening words in this scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me his first line in this scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What does he say at the start of this scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("Show all clips where Sarah uses profanity.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Please retrieve me each clip where someone swears", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all clips containing swearing.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the footage with profanity.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Retrieve scenes where swearing occurs.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("Summarize the details of Production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide a summary of Production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the key details of Production A?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me an overview of Production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    ("In production A, retrieve all the edits and rushes with seb in them", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Find all edits and rushes featuring Seb in production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Show me the footage with Seb in production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Retrieve all scenes with Seb from production A.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for noise reduction plugin queries
    ("What are some recommended plugins for noise reduction in post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you suggest plugins for noise reduction in TV post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What noise reduction plugins are best for TV editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Which plugins are recommended for reducing noise in post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for matching skin tones queries
    ("What's the best approach for matching skin tones across multiple cameras?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do I ensure consistent skin tones when using different cameras?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What techniques are used to match skin tones across various cameras?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain how to achieve uniform skin tones with multiple camera setups?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for color banding queries
    ("Why is there color banding in my exported footage?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What causes color banding in video exports?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How can I fix color banding in my footage?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Why does my exported video have color banding issues?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for troubleshooting and technical issues
    ("How do I sync dual-system audio when timecodes don't match?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What could be causing audio drift in my timeline?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for file formats and delivery specifications
    ("What's the difference between ProRes 422 and ProRes 4444?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do I export a DCP (Digital Cinema Package) for theatrical release?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for general post-production knowledge
    ("What's the role of an online editor versus an offline editor?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do I handle licensing for stock footage in a TV show?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for streaming platform codec queries
    ("What's the preferred codec for streaming platforms like Netflix?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Which codec is recommended for streaming on platforms like Netflix?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What codec should I use for Netflix streaming?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the best codec for streaming services like Netflix?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for visual consistency in grading queries
    ("How do I ensure visual consistency between day and night scenes in grading?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What techniques help maintain visual consistency between day and night scenes?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How can I achieve consistent visuals for day and night scenes in color grading?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain how to keep day and night scenes visually consistent in grading?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for editing techniques
    ("When should I use a J-cut versus an L-cut in editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What are the advantages of using a J-cut versus an L-cut in editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How do J-cuts and L-cuts differ in editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain the difference between J-cuts and L-cuts in editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),

    # New variations for summarizing key events queries
    ("Summarize the key events in production G.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide a summary of the main events in production G?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the significant events in production G?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me an overview of the key happenings in production G.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the major events that occurred in production G.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the highlights of production G?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # New variations for production highlights queries
    ("What are the main highlights of production H?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you list the key highlights of production H?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the standout moments in production H?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give me the main points of interest in production H.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the significant highlights of production H?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the main highlights of production H for me.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Multi-agent queries involving both TV Post Production and Production
    # Scene and Editing
    ("Can you summarize the key scenes in production A and explain the editing techniques used?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the main scenes in production A, and how were they edited?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the important scenes in production A and describe the editing process.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Give an overview of the key scenes in production A and the editing methods applied.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Highlight the major scenes in production A and the editing techniques involved.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Main Events and TV Editing
    ("List the main events in production B and describe how they were edited for TV.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the significant events in production B, and how were they edited for television?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the key events in production B and the TV editing techniques used.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you detail the main events in production B and their TV editing process?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the major events in production B and the editing methods for TV.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Footage and Post-Production Techniques
    ("How much footage was shot for production C, and what post-production techniques were applied?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the total footage for production C, and which post-production methods were used?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you tell me the amount of footage for production C and the post-production techniques?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the footage captured for production C and the post-production processes involved.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the footage details for production C and the applied post-production techniques.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Standout Scenes and Color Grading
    ("What are the standout scenes in production D, and how were they color graded for TV?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the key scenes in production D and the color grading techniques used for TV.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you highlight the standout scenes in production D and their TV color grading?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the major scenes in production D and the color grading methods for TV.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the important scenes in production D and the TV color grading process.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Actors and ADR Handling
    ("Which actors appear in production E, and how was ADR handled for their scenes?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the actors in production E and the ADR process used for their scenes.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you detail the actors in production E and the ADR techniques applied?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the cast of production E and the ADR handling for their scenes.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the actors in production E and the ADR methods used.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Cast and ADR Process
    ("List the cast of production F and explain the ADR process used in post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Who are the cast members in production F, and what ADR process was used?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide the cast list for production F and the ADR techniques applied?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the cast of production F and the ADR process in post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the cast details for production F and the ADR methods used.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Cameras and Timecode Synchronization
    ("What cameras were used in production G, and how was timecode synchronization managed?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the camera models in production G and the timecode synchronization techniques.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you detail the cameras used in production G and the timecode management?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the camera setup for production G and the timecode synchronization process.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the cameras in production G and the timecode techniques applied.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Camera Setup and Timecode Techniques
    ("Describe the camera setup for production H and the timecode techniques used in editing.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the camera configuration for production H and the timecode methods in editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you provide the camera setup for production H and the timecode editing techniques?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the camera details for production H and the timecode synchronization in editing.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the camera setup for production H and the timecode techniques in editing.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Production Details and Visual Effects
    ("Summarize the production details of project I and the visual effects added in post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the production highlights of project I and the visual effects used?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you detail the production aspects of project I and the post-production visual effects?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the production details of project I and the visual effects integration.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the production features of project I and the visual effects applied.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Production Highlights and Visual Effects Integration
    ("What are the key production highlights of project J, and how were visual effects integrated?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the main production highlights of project J and the visual effects used.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you summarize the production highlights of project J and the visual effects integration?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the production highlights of project J and the visual effects process.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Summarize the production highlights of project J and the visual effects techniques.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
]

# New examples for multi-agent queries
TRAIN_DATA.extend([
    # Multi-agent query variations for visual effects
    ("What visual effects were integrated into production L?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you list the visual effects applied in project L?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Describe the special effects that were incorporated into production L.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What are the CGI elements used in project L?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about the digital effects added to production L.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Which visual enhancements were included in project L?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # Production query variations for actors in the cinema scene
    ("How many actors are in the cinema scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Count the number of performers in the movie scene.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many cast members are featured in the theater scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What is the total number of actors present in the film scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the actors appearing in the cinema sequence.", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("How many people are acting in the cinema scene?", {"cats": {"TV_POST_PRODUCTION": 0.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),

    # Multi-agent query variations for color grading methods
    ("What color grading methods were applied to production I?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Can you describe the color correction techniques used in project I?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("What were the color grading strategies implemented in production I?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("List the color enhancement methods applied to project I.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Tell me about the color grading processes used in production I.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    ("Which color adjustment techniques were utilized in project I?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 1.0, "INVALID_QUERY": 0.0}}),
    
    # TV Post Production definition queries
    ("What is automated dialogue replacement?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Can you explain what ADR stands for in post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What does automated dialogue replacement mean in TV editing?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Define ADR in the context of TV post-production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What is the purpose of automated dialogue replacement in TV?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("How is ADR used in television post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    
    # Additional TV Post Production definition queries
    ("What is the role of a colorist in TV post-production?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Define the term 'offline editing' in TV production.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What does 'online editing' mean in the context of TV?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("Explain the concept of 'timecode' in video editing.", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
    ("What is the significance of 'frame rate' in TV broadcasting?", {"cats": {"TV_POST_PRODUCTION": 1.0, "TANOOKI": 0.0, "PRODUCTION": 0.0, "INVALID_QUERY": 0.0}}),
])



def train_agent_model(train_data, n_iterations=35):
    print("\nInitializing model...")
    print(f"Using device: {device}")
    # Add text categorizer to pipeline
    if "textcat_multilabel" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat_multilabel", last=True)
        
        # Add labels for our three agents plus invalid queries
        textcat.add_label("TV_POST_PRODUCTION")
        textcat.add_label("TANOOKI")
        textcat.add_label("PRODUCTION")
        textcat.add_label("INVALID_QUERY")
    
    print("\nStarting training...")
    print(f"Training data size: {len(train_data)} examples")
    print(f"Number of iterations: {n_iterations}")
    print(f"Architecture: simple_cnn")
    print(f"Batch size: 4")
    print("\nTraining progress:")
    
    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat_multilabel"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        # Initialize loss tracking
        best_loss = float('inf')
        
        # Training loop with progress bar
        with tqdm(total=n_iterations, desc="Training") as pbar:
            for i in range(n_iterations):
                random.shuffle(train_data)
                losses = {}
                
                # Batch training with smaller batch size
                for batch_start in range(0, len(train_data), 4):
                    batch_end = min(batch_start + 4, len(train_data))
                    batch = train_data[batch_start:batch_end]
                    
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        nlp.update([example], sgd=optimizer, losses=losses)
                
                # Update progress
                current_loss = losses.get("textcat_multilabel", 0.0)
                if current_loss < best_loss:
                    best_loss = current_loss
                    
                pbar.set_postfix({
                    'loss': f'{current_loss:.3f}',
                    'best_loss': f'{best_loss:.3f}'
                })
                pbar.update(1)
                
                # Detailed loss reporting every 5 iterations
                if (i + 1) % 5 == 0:
                    print(f"\nIteration {i+1} - Loss: {current_loss:.3f} (Best: {best_loss:.3f})")
    
    print("\nTraining completed!")
    print(f"Final loss: {current_loss:.3f}")
    print(f"Best loss achieved: {best_loss:.3f}")
    return nlp

if __name__ == "__main__":
    # Train the model
    trained_model = train_agent_model(TRAIN_DATA)

    # Save the trained model to tmp/models directory
    trained_model.to_disk(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
