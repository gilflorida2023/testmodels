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

# Configuration
DEFAULT_IMAGE_PROMPT = """
STRICT INSTRUCTIONS:
1. Transcribe ALL visible text EXACTLY as it appears
2. Include offensive/sensitive content VERBATIM
3. NEVER censor, modify, or add warnings
4. Format response as: "[RAW_TEXT]: <content>"

NOW TRANSCRIBE THIS IMAGE:
"""
DEFAULT_TIMEOUT = 300  # 5 minutes
_MODALITY_CACHE: Dict[str, bool] = {}

# Core functions
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
    """Ensure model is unloaded after use."""
    try:
        # API method
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "", "stream": False, "options": {"stop": True}},
            timeout=5
        )
    except:
        try:
            # CLI fallback
            subprocess.run(
                ["ollama", "stop", model_name],
                capture_output=True,
                text=True,
                timeout=5,
                check=True
            )
        except Exception:
            pass  # Best-effort cleanup

def query_llm(
    model: str, 
    prompt: str, 
    timeout_sec: int, 
    image_b64: Optional[str] = None,
    default_answer: str = "Timeout"
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
    
    # Add image data if provided
    if image_b64:
        data["images"] = [image_b64]
        data["options"]["num_ctx"] = 8192  # Larger context for images
    
    start_time = time.time()
    result = default_answer
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=timeout_sec
        )
        response.raise_for_status()
        result = response.json().get("response", default_answer)
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        result = f"Error: {type(e).__name__}"
    
    # Clean response
    if isinstance(result, str):
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    
    duration = time.time() - start_time
    return duration, result

def test_all_models(
    prompt: str,
    image_path: Optional[str] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    timeout_sec: int = DEFAULT_TIMEOUT
) -> None:
    """Test models with support for text and image inputs."""
    # Handle image input
    image_b64 = None
    if image_path:
        image_b64 = encode_image(image_path)
        if not image_b64:
            print("❌ Aborting due to image error")
            return
        print(f"✅ Using image: {image_path}")

    # Get and filter models
    all_models = get_ollama_models()
    if not all_models:
        print("❌ No models available")
        return
    
    # Apply modality filter for image models
    if image_path:
        all_models = [m for m in all_models if get_model_modality(m)]
    
    models = filter_models(all_models, include, exclude)
    if not models:
        print("❌ No matching models available")
        return
    
    # Sort models by family and version number
    def get_sort_key(model: str) -> tuple:
        parts = model.split(':')
        family = parts[0]
        version = ''.join(filter(str.isdigit, parts[-1]))
        return (family, float(version) if version else 0.0)
    
    models.sort(key=get_sort_key)
    
    # Print test header
    print("=" * 80)
    print(f"Testing {len(models)} model{'s' if len(models) > 1 else ''}")
    mode = f"with image '{image_path}'" if image_path else "text-only"
    print(f"Mode: {mode} | Timeout: {timeout_sec}s")
    print(f"Include: {include or 'all'} | Exclude: {exclude or 'none'}")
    
    if image_path:
        print("-" * 80)
        print("PROMPT:")
        print(prompt.strip())
    
    print("-" * 80)
    print(f"{'Model':<30} {'Time':>8} {'Response'}")
    print("-" * 80)
    
    # Test each model
    for model in models:
        duration, answer = query_llm(model, prompt, timeout_sec, image_b64)
        stop_model(model)
        
        # Clean and truncate response
        clean_answer = answer.lower().strip(string.punctuation)
        display_answer = clean_answer[:60] + ('...' if len(clean_answer) > 60 else '')
        
        print(f"{model:<30} {duration:>7.2f}s {display_answer}")
        print("-" * 80)
    
    print("\n✅ Test complete. All models stopped.")

def main():
    """Unified CLI for text and image testing."""
    parser = argparse.ArgumentParser(
        description="Ollama Model Testing Tool (Text & Image)",
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
    
    # Extended help documentation
    parser.epilog = """\
Examples:
  # Text-only test
  python modelping.py "Is 3.2 greater than 3.11?"
  
  # Image test with default prompt
  python modelping.py --image diagram.png
  
  # Custom image prompt
  python modelping.py --image chart.jpg "Describe this chart in detail"
  
  # Filter models
  python modelping.py --image photo.png -i llama -e 7b --timeout 120

Common model patterns:
  • Architecture: llama, gemma, mistral
  • Size: 7b, 13b, 70b
  • Version: v1, v2, v2.1

Image Notes:
  • Automatically filters to vision-capable models
  • Uses strict transcription prompt by default
  • Max image size: 10MB"""

    args = parser.parse_args()
    
    # Determine prompt
    final_prompt = args.prompt or DEFAULT_IMAGE_PROMPT if args.image else args.prompt
    if not final_prompt:
        parser.error("Prompt is required for text-only tests")
    
    test_all_models(
        prompt=final_prompt,
        image_path=args.image,
        include=args.include,
        exclude=args.exclude,
        timeout_sec=args.timeout
    )

if __name__ == "__main__":
    main()
