import requests
import time
import json

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

def query_llm_with_timeout(model, prompt, timeoutsecs, answer="Timeout"):
    """
    Query an LLM with a timeout and return execution duration and answer.
    
    Args:
        model (str): The model to use for generation
        prompt (str): The prompt to send to the model
        timeoutsecs (int): Maximum allowed time in seconds
        answer (str): Default answer to return if timeout occurs
        
    Returns:
        tuple: (execution_duration, answer)
               execution_duration is in seconds
               answer is the complete response or the default answer if timeout occurred
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    start_time = time.time()
    result = answer  # Initialize with default answer
    
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
        #print(f"Timeout reached after {timeoutsecs} seconds")
    except Exception as e:
        pass
        #print(f"Error querying model {model}: {e}")
    
    execution_duration = time.time() - start_time
    return (execution_duration, result.strip() if result else answer)

def test_all_models():
    """
    Test all available models with a standard prompt and timeout
    """
    #prompt = "how many r's in strawberry?"
    prompt = "one word response, yes or no, is 3.2 >  3.11?"
    timeout_sec = 200  # 5 minutes timeout
    
    #models = get_ollama_models()
    # Get and sort models in one line
    models = sorted(get_ollama_models())
    
    if not models:
        print("No models available to test")
        return
    
    print(f"Testing {len(models)} models with prompt: '{prompt}' \ntimeout: '{timeout_sec}' seconds.")
    print("=" * 60)
    
    for model in models:
        print(f"Testing model: {model}")
        duration, answer = query_llm_with_timeout(model, prompt, timeout_sec)
        
        print(f"Execution duration: {duration:.2f} seconds")
        print(f"Answer: {answer}")
        print("=" * 60)

if __name__ == "__main__":
    test_all_models()
