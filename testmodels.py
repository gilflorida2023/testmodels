import requests
import time
import json
import re
import string 
import subprocess
import sys
import argparse
from typing import Tuple, List, Optional

def get_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return []

def filter_models(models: List[str], include: Optional[str], exclude: Optional[str]) -> List[str]:
    """Filter models based on include/exclude patterns."""
    if include:
        include_patterns = [p.strip().lower() for p in include.split(",")]
        models = [m for m in models if any(p in m.lower() for p in include_patterns)]
    
    if exclude:
        exclude_patterns = [p.strip().lower() for p in exclude.split(",")]
        models = [m for m in models if not any(p in m.lower() for p in exclude_patterns)]
    
    return models

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
        pass
    except requests.exceptions.RequestException as e:
        pass
    except json.JSONDecodeError:
        pass
    
    # Clean response
    if isinstance(result, str):
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    
    duration = time.time() - start_time
    return (duration, result if result else default_answer)

def test_all_models(prompt: str, include: str = None, exclude: str = None, 
                   timeout_sec: int = 300) -> None:
    """Test all models with proper cleanup and filtering."""
    models = get_ollama_models()
    if not models:
        print("No models available to test")
        return
    
    # Apply filters
    models = filter_models(models, include, exclude)
    if not models:
        print("❌ No matching models available")
        return
    
    # Sort models by family and version
    models.sort(key=lambda x: (x.split(':')[0], extract_model_number(x)))
    
    print("=" * 80)
    print(f"Testing {len(models)} model{'s' if len(models) > 1 else ''}")
    print(f"Prompt: {prompt}")
    print(f"Timeout: {timeout_sec}s | Include: {include or 'all'} | Exclude: {exclude or 'none'}")
    print("-" * 80)
    print(f"{'Model':<30} {'Time':>8} {'Response'}")
    print("-" * 80)
    
    for model in models:
        duration, answer = query_llm_with_timeout(model, prompt, timeout_sec)
        stop_model(model)  # Ensure model is stopped
        
        # Clean and truncate response for display
        answer_clean = answer.lower().strip(string.punctuation)
        answer_display = (answer_clean[:60] + '...') if len(answer_clean) > 60 else answer_clean
        
        print(f"{model:<30} {duration:>7.2f}s {answer_display}")
        print("-" * 80)
    
    print("\n✅ Test complete. All models stopped.")

def main():
    """Command-line interface with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Ollama Model Testing Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="The prompt to test models with (can also use --prompt)"
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Custom prompt to test models with (overrides positional prompt)"
    )
    parser.add_argument(
        "-i", "--include",
        help="Comma-separated patterns to include models (e.g. 'llama,7b')",
        metavar="PATTERNS"
    )
    parser.add_argument(
        "-e", "--exclude",
        help="Comma-separated patterns to exclude models (e.g. 'llava,13b')",
        metavar="PATTERNS"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout per model in seconds (default: 300 = 5 min)"
    )
    
    # Extended help documentation
    parser.epilog = """\
Examples:
  # Basic usage with positional prompt
  python testmodels.py "Is 3.2 greater than 3.11?"
  
  # Custom prompt with filtering
  python testmodels.py --prompt "Explain quantum computing" --include llama,7b
  
  # Exclude specific model types
  python testmodels.py "What is AI?" --exclude mistral,13b --timeout 120
  
  # Combine include and exclude filters
  python testmodels.py "2+2=?" -i llama -e 70b -t 30

Common model patterns:
  • Architecture: llama, gemma, mistral, fuyu
  • Size: 7b, 13b, 70b
  • Version: v1, v2, v2.1, latest

Note: Filters use case-insensitive substring matching"""

    # Parse arguments
    args = parser.parse_args()
    
    # Determine prompt source
    final_prompt = args.prompt if args.prompt is not None else args.prompt
    if not final_prompt:
        parser.error("Prompt is required. Provide it as a positional argument or with --prompt")
    
    test_all_models(
        prompt=final_prompt,
        include=args.include,
        exclude=args.exclude,
        timeout_sec=args.timeout
    )

if __name__ == "__main__":
    main()
