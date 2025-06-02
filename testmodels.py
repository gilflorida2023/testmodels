import requests
import time
import json
import re

def get_ollama_models():
    """
    Get a list of available Ollama model names.
    
    Returns:
        list: A list of model name strings
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model['name'] for model in data.get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def extract_model_number(model_name):
    """
    Extract the numeric part from model names for proper sorting.
    Handles formats like 'gemma3:1b', 'gemma3:12b', 'deepseek-r1:1.5b', etc.
    """
    # Split on ':' and take the last part
    version_part = model_name.split(':')[-1]
    # Extract all numbers and decimals
    numbers = re.findall(r'\d+\.?\d*', version_part)
    if numbers:
        return float(numbers[0])
    return 0

def query_llm_with_timeout(model, prompt, timeoutsecs, answer="Timeout"):
    """
    Query an LLM with a timeout and return execution duration and answer.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    start_time = time.time()
    result = answer
    
    try:
        with requests.post(url, json=data, stream=True, timeout=timeoutsecs) as r:
            for line in r.iter_lines():
                if line and time.time() - start_time < timeoutsecs:
                    json_chunk = json.loads(line.decode())
                    if "response" in json_chunk:
                        result = json_chunk["response"]
                else:
                    raise TimeoutError("Timeout reached")
    except TimeoutError:
        pass
    except Exception as e:
        pass
    
    if result and isinstance(result, str):
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    
    execution_duration = time.time() - start_time
    return (execution_duration, result if result else answer)

def test_all_models():
    """
    Test all available models with a standard prompt and timeout
    """
    prompt = "No explanation. Brief one word response, yes or no.Question: is 3.2 >  3.11?"
    timeout_sec = 100
    
    models = get_ollama_models()
    # Custom sorting: first by model family, then by extracted version number
    models.sort(key=lambda x: (x.split(':')[0], extract_model_number(x)))
    
    if not models:
        print("No models available to test")
        return
    
    print(f"Testing {len(models)} models with prompt: '{prompt}' \ntimeout: '{timeout_sec}' seconds.")
    print("=" * 60)
    
    for model in models:
        duration, answer = query_llm_with_timeout(model, prompt, timeout_sec)
        #print(f"Testing model: {model}")
        #print(f"Execution duration: {duration:.2f} seconds")
        #print(f"Answer: {answer}")
        print(f"{model},{duration:.2f} seconds,{answer}")

if __name__ == "__main__":
    test_all_models()
