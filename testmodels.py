import requests
import time
import json
import re
import string 
import subprocess
from typing import Tuple, List

def get_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def extract_model_number(model_name: str) -> float:
    """Extract version number for sorting."""
    version_part = model_name.split(':')[-1]
    numbers = re.findall(r'\d+\.?\d*', version_part)
    return float(numbers[0]) if numbers else 0

def stop_model(model_name: str) -> None:
    """
    Try to stop model using API first, fall back to CLI if needed.
    """
    # First attempt: Try API method
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "",
                "stream": False,
                "options": {"stop": True}
            },
            timeout=5
        )
        if response.status_code == 200:
            return  # API method worked
    except requests.exceptions.RequestException:
        pass  # We'll fall back to CLI
    
    # Fallback: Use CLI command
    try:
        result = subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"Warning: Could not stop {model_name}. Error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"Timeout while trying to stop {model_name}")
    except Exception as e:
        print(f"Error stopping {model_name}: {str(e)}")


def query_llm_with_timeout(model: str, prompt: str, timeout_sec: int, default_answer: str = "Timeout") -> Tuple[float, str]:
    """Query LLM with strict timeout and guaranteed cleanup."""
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    
    start_time = time.time()
    result = default_answer
    
    try:
        response = requests.post(url, json=data, timeout=timeout_sec)
        response.raise_for_status()
        result = response.json().get("response", default_answer)
    except requests.exceptions.Timeout:
        pass #print(f"Timeout reached for {model}")
    except requests.exceptions.RequestException as e:
        pass #print(f"Request error for {model}: {e}")
    except json.JSONDecodeError:
        pass #print(f"Invalid response from {model}")
    finally:
        #stop_model(model)  # Ensure model is stopped
        pass
    
    # Clean response
    if isinstance(result, str):
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    
    duration = time.time() - start_time
    return (duration, result if result else default_answer)

def test_all_models() -> None:
    """Test all models with proper cleanup."""
    prompt = "Strictly answer only yes or no with no additional words, punctuation, or explanation.  Question: Is 3.2 > 3.11?"
    timeout_sec = 60  # More reasonable timeout
    
    models = get_ollama_models()
    if not models:
        print("No models available to test")
        return
    
    # Sort models by family and version
    models.sort(key=lambda x: (x.split(':')[0], extract_model_number(x)))
    
    print(f"prompt:{prompt}")
    print(f"Testing {len(models)} models (timeout: {timeout_sec}s)")
    print("=" * 60)
    print(f"{'Model':<30} {'Time':>7} {'Response':<10}")
    print("-" * 60)
    
    for model in models:
        duration, answer = query_llm_with_timeout(model, prompt, timeout_sec)
        stop_model(model)  # Ensure model is stopped
        # Convert to lowercase and strip punctuation from the start/end
        answer_clean = answer.lower().strip(string.punctuation)
        print(f"{model:<30} {duration:>7.2f}s {answer_clean:<10}")
    
    print("\nTest complete. All models stopped.")

if __name__ == "__main__":
    test_all_models()
