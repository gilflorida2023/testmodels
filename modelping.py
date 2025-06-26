#!/usr/bin/env python3
import requests
import time
import json
import re
import string
import subprocess
import sys
import argparse
import base64
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from statistics import mean, stdev
import csv
from datetime import datetime

# Configuration
DEFAULT_IMAGE_PROMPT = """
Extract ALL visible text EXACTLY as it appears in the image.
Include ALL content verbatim without modification.
Respond with ONLY the raw text content.
"""
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_REPEATS = 1
_MODALITY_CACHE: Dict[str, bool] = {}

# JSON Schemas for response formatting
IMAGE_TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "raw_text": {
            "type": "string",
            "description": "The exact transcription of all visible text"
        }
    },
    "required": ["raw_text"]
}

NUMERIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "number",
            "description": "The numeric answer to the question"
        },
        "error": {
            "type": "string",
            "description": "Error message if unable to provide numeric answer"
        }
    },
    "required": ["answer"]
}

class TestResult:
    def __init__(self, model: str):
        self.model = model
        self.responses = []
        self.durations = []
        self.correct = 0
    
    def add_run(self, response: str, duration: float, expected: Optional[str] = None):
        self.responses.append(response)
        self.durations.append(duration)
        if expected:
            self.correct += int(self._normalize(response) == self._normalize(expected))
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        return text.lower().strip().translate(str.maketrans('', '', string.punctuation))
    
    @property
    def accuracy(self) -> float:
        return self.correct / len(self.responses) if self.responses else 0.0
    
    @property
    def avg_duration(self) -> float:
        return mean(self.durations) if self.durations else 0.0
    
    @property
    def stdev_duration(self) -> float:
        return stdev(self.durations) if len(self.durations) > 1 else 0.0

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

def get_model_modality(model_name: str) -> bool:
    """Check if model supports images."""
    if model_name in _MODALITY_CACHE:
        return _MODALITY_CACHE[model_name]
    
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=10
        )
        response.raise_for_status()
        is_multimodal = "vision" in response.json().get("capabilities", [])
        _MODALITY_CACHE[model_name] = is_multimodal
        return is_multimodal
    except Exception:
        # Fallback heuristic
        multimodal_keywords = ["vision", "llava", "bakllava", "fuyu", "cogvlm"]
        is_multimodal = any(kw in model_name.lower() for kw in multimodal_keywords)
        _MODALITY_CACHE[model_name] = is_multimodal
        return is_multimodal

def get_filtered_models(require_vision: bool, include: Optional[str], exclude: Optional[str]) -> List[str]:
    """Get models with optional filtering."""
    models = get_ollama_models()
    if not models:
        print("❌ No models available")
        sys.exit(1)
    
    if require_vision:
        models = [m for m in models if get_model_modality(m)]
    
    models = filter_models(models, include, exclude)
    if not models:
        print("❌ No matching models available")
        sys.exit(1)
    
    # Sort models by family and version number
    def get_sort_key(model: str) -> tuple:
        parts = model.split(':')
        family = parts[0]
        version = ''.join(filter(str.isdigit, parts[-1]))
        return (family, float(version) if version else 0.0)
    
    models.sort(key=get_sort_key)
    return models

def encode_image(image_path: str) -> Optional[str]:
    """Convert image to base64 with validation."""
    try:
        path = Path(image_path)
        if not path.exists():
            print(f"❌ Image not found: {image_path}")
            return None
        if path.stat().st_size > 10 * 1024 * 1024:  # 10MB
            print(f"❌ Image too large (max 10MB): {image_path}")
            return None
            
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Image error: {e}")
        return None

def stop_model(model_name: str) -> None:
    """Try to stop model using API first, fall back to CLI if needed."""
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
            return
    except requests.exceptions.RequestException:
        pass
    
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

def query_llm(
    model: str, 
    prompt: str, 
    timeout_sec: int, 
    image_b64: Optional[str] = None,
    default_answer: str = "Timeout",
    response_schema: Optional[Dict] = None
) -> Tuple[float, str]:
    """Query LLM with support for text and image inputs."""
    data: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "top_k": 1
        }
    }
    
    if image_b64:
        data["images"] = [image_b64]
        data["options"]["num_ctx"] = 8192
    
    if response_schema:
        data["options"]["format"] = "json"
        data["schema"] = response_schema
    
    start_time = time.time()
    result = default_answer
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=timeout_sec
        )
        response.raise_for_status()
        json_response = response.json()
        
        if response_schema:
            try:
                if isinstance(json_response.get("response"), str):
                    # First try to parse as JSON
                    try:
                        parsed = json.loads(json_response["response"])
                        if "answer" in parsed:  # Numeric answer case
                            result = str(parsed["answer"])
                        elif "raw_text" in parsed:  # Image case
                            result = parsed["raw_text"]
                        elif "error" in parsed:
                            result = f"Error: {parsed['error']}"
                        else:
                            result = "Unexpected JSON format"
                    except json.JSONDecodeError:
                        # If not JSON, try to extract number from plain text
                        numbers = re.findall(r'\d+', json_response["response"])
                        if numbers:
                            result = numbers[0]
                        else:
                            result = "No numeric value found"
            except Exception as e:
                result = f"Parse error: {str(e)}"
        else:
            result = json_response.get("response", default_answer)
            
    except requests.exceptions.Timeout:
        result = "Timeout"
    except Exception as e:
        result = f"Error: {type(e).__name__}"
    finally:
        stop_model(model)
    
    if isinstance(result, str):
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    
    duration = time.time() - start_time
    return duration, result

def run_test_cycle(
    models: List[str],
    prompt: str,
    repeats: int,
    image_b64: Optional[str],
    timeout: int,
    expected_answer: Optional[str],
    response_schema: Optional[Dict] = None
) -> Dict[str, TestResult]:
    """Run tests for all models with multiple repetitions"""
    results = {model: TestResult(model) for model in models}
    
    for iteration in range(1, repeats + 1):
        print(f"prompt: {prompt}")
        print(f"\n=== Iteration {iteration}/{repeats} ===")
        for model in models:
            duration, response = query_llm(
                model, prompt, timeout, image_b64, response_schema=response_schema
            )
            results[model].add_run(response, duration, expected_answer)
            
            # Display interim results
            print(f"{model:<30} {duration:>7.2f}s | ", end='')
            if expected_answer:
                print(f"Acc: {results[model].accuracy*100:.1f}%", end='')
            response_display = response.replace('\n', ' ').replace('\r', ' ')
            if len(response_display) > 60:
                response_display = response_display[:57] + "..."
            print(f" | {response_display}")
            
        if iteration < repeats:
            time.sleep(2)
    
    return results

def save_results(
    results: Dict[str, TestResult],
    prompt: str,
    image_path: Optional[str],
    expected: Optional[str],
    output_format: str = "both"
):
    """Save results in CSV and/or JSON format"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"llm_benchmark_{timestamp}"
    
    summary = {
        "metadata": {
            "timestamp": timestamp,
            "prompt": prompt,
            "image": image_path,
            "expected_answer": expected,
            "num_repeats": len(next(iter(results.values())).responses) if results else 0
        },
        "results": {
            model: {
                "responses": result.responses,
                "durations": result.durations,
                "accuracy": result.accuracy,
                "avg_duration": result.avg_duration,
                "stdev_duration": result.stdev_duration
            }
            for model, result in results.items()
        }
    }
    
    if output_format in ("json", "both"):
        with open(f"{base_name}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved JSON results to {base_name}.json")
    
    if output_format in ("csv", "both"):
        with open(f"{base_name}.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Avg Duration", "StdDev", 
                "Accuracy", "Responses"
            ])
            for model, result in results.items():
                writer.writerow([
                    model,
                    result.avg_duration,
                    result.stdev_duration,
                    result.accuracy,
                    " | ".join(f'"{r}"' for r in result.responses)
                ])
        print(f"✓ Saved CSV results to {base_name}.csv")

def main():
    parser = argparse.ArgumentParser(
        description="LLM Benchmarking Tool with Statistical Analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="The prompt to test models with"
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Custom prompt (overrides positional prompt)"
    )
    parser.add_argument(
        "-img", "--image",
        help="Path to image file for multimodal testing"
    )
    parser.add_argument(
        "-i", "--include",
        help="Comma-separated patterns to include models (e.g. 'llama,7b')"
    )
    parser.add_argument(
        "-e", "--exclude",
        help="Comma-separated patterns to exclude models (e.g. 'llava,13b')"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per model in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of test repetitions (default: {DEFAULT_REPEATS})"
    )
    parser.add_argument(
        "-exp", "--expected",
        help="Expected answer for accuracy calculation"
    )
    parser.add_argument(
        "-o", "--output",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format(s) for results"
    )
    parser.add_argument(
        "-s", "--schema",
        choices=["image", "numeric", "none"],
        default="none",
        help="Response schema type (image: for text extraction, numeric: for number responses)"
    )
    
    parser.epilog = """\
Examples:
  # Basic text test with 3 repetitions
  python modelping.py "2+2=?" -exp "4" -r 3 -s numeric
  
  # Image test with accuracy checking
  python modelping.py --image diagram.png -exp "42" -r 5 -s image
  
  # Force numeric output for counting
  python modelping.py "How many r's in 'strawberry'?" -s numeric -exp "3"
  
  # Filter models and save JSON only
  python modelping.py "Capital of France" -exp "paris" -i llama -o json
  
  # Full benchmark with all features
  python modelping.py --image chart.jpg "Describe this chart" \\
    -exp "sales chart" -i llama,7b -e llava -r 10 -t 120 -s image"""

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    if not args.image and not args.prompt and not args.prompt:
        parser.error("Prompt is required for text-only tests")
    
    final_prompt = args.prompt or args.prompt or DEFAULT_IMAGE_PROMPT
    image_b64 = encode_image(args.image) if args.image else None
    
    response_schema = None
    if args.schema == "image":
        response_schema = IMAGE_TRANSCRIPTION_SCHEMA
        final_prompt = "Extract all visible text exactly as it appears in the image."
    elif args.schema == "numeric":
        response_schema = NUMERIC_RESPONSE_SCHEMA
        final_prompt = (
            f"{final_prompt}\n"
            "You must respond with a JSON object containing a numeric 'answer' field.\n"
            "Example: {{\"answer\": 42}}"
        )
    
    models = get_filtered_models(
        require_vision=args.image is not None,
        include=args.include,
        exclude=args.exclude
    )
    
    print(f"\n🚀 Testing {len(models)} model{'s' if len(models) > 1 else ''} "
          f"with {args.repeats} repetition{'s' if args.repeats > 1 else ''}")
    if args.image:
        print(f"📷 Image: {args.image}")
    print(f"⏱  Timeout: {args.timeout}s")
    if args.expected:
        print(f"🎯 Expected answer: '{args.expected}'")
    if args.schema != "none":
        print(f"📝 Response schema: {args.schema}")
    
    results = run_test_cycle(
        models=models,
        prompt=final_prompt,
        repeats=args.repeats,
        image_b64=image_b64,
        timeout=args.timeout,
        expected_answer=args.expected,
        response_schema=response_schema
    )
    
    print("\n=== FINAL RESULTS ===")
    print(f"{'MODEL':<30} {'AVG TIME':>10} {'ACCURACY':>10} {'STDEV':>10}")
    for model, result in results.items():
        print(f"{model:<30} {result.avg_duration:>9.2f}s {result.accuracy*100:>9.1f}% {result.stdev_duration:>9.2f}s")
    
    save_results(
        results=results,
        prompt=final_prompt,
        image_path=args.image,
        expected=args.expected,
        output_format=args.output
    )

if __name__ == "__main__":
    main()
