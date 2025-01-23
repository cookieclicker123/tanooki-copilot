import requests
import json

def keep_model_loaded(model_name):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": "Hello",
        "keep_alive": "24h",
        "context_size": 4096,
        "stream": False  # This tells the API to not stream the response
    }
    response = requests.post(url, json=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            try:
                json_response = json.loads(line)
                print(json_response)
            except json.JSONDecodeError:
                print(f"Could not parse line: {line}")

# Replace "tv_model2" with your model name
keep_model_loaded("tv_model2:latest")
