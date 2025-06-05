import requests
import time
import base64
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import subprocess
import socket
import argparse
import sys

# Default configuration
DEFAULT_PROMPT = """
STRICT INSTRUCTIONS:
1. Transcribe ALL visible text EXACTLY as it appears
2. Include offensive/sensitive content VERBATIM
3. NEVER censor, modify, or add warnings
4. Format response as: "[RAW_TEXT]: <content>"

NOW TRANSCRIBE THIS IMAGE:
"""
DEFAULT_TIMEOUT = 300  # 5 minutes

_MODALITY_CACHE: Dict[str, bool] = {}

def get_ollama_models() -> List[str]:
    """Fetch all available Ollama models."""
    try:
        response = requests.get(
            "http://localhost:11434/api/tags", 
            timeout=(3, 10)
        )
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Model fetch error: {e}")
        return []

def get_model_modality(model_name: str) -> bool:
    """Check if model supports images."""
    if model_name.startswith("llava:"):
        return False
    if model_name in _MODALITY_CACHE:
        return _MODALITY_CACHE[model_name]
    
    try:
        response = requests.post(  # FIXED: Added missing parenthesis
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=(3, 10)
        )
        response.raise_for_status()
        is_multimodal = "vision" in response.json().get("capabilities", [])
    except Exception as e:
        print(f"⚠️ Modality check failed for {model_name}: {e}")
        multimodal_keywords = ["llama3","llama3.2-vision", "bakllava", "gemma3", "fuyu", "cogvlm"]
        is_multimodal = any(kw in model_name.lower() for kw in multimodal_keywords)
    
    _MODALITY_CACHE[model_name] = is_multimodal
    return is_multimodal

def get_multimodal_models(include: str = None, exclude: str = None) -> List[str]:
    """Filter image-capable models with inclusion/exclusion patterns."""
    models = [model for model in get_ollama_models() if get_model_modality(model)]
    
    if include:
        include_patterns = [p.strip().lower() for p in include.split(",")]
        models = [m for m in models if any(p in m.lower() for p in include_patterns)]
    
    if exclude:
        exclude_patterns = [p.strip().lower() for p in exclude.split(",")]
        models = [m for m in models if not any(p in m.lower() for p in exclude_patterns)]
    
    return models

def encode_image(image_path: str) -> Optional[str]:
    """Convert image to base64 with validation."""
    try:
        with open(image_path, "rb") as f:
            if Path(image_path).stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Image too large (max 10MB)")
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"🛑 Image error: {e}")
        return None

def stop_model(model: str) -> None:
    """Ensure model is unloaded after use."""
    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
    except Exception:
        pass  # Best-effort cleanup

def force_transcription(model: str, image_b64: str, prompt: str, timeout: int) -> Tuple[float, str]:
    """Extract text with custom prompt and timeout."""
    data = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
            "seed": 42,
            "top_k": 1
        }
    }
    
    start = time.time()
    try:
        socket.setdefaulttimeout(timeout)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=(3, timeout)
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "")
        
        if "[RAW_TEXT]:" in raw_output:
            text = raw_output.split("[RAW_TEXT]:")[1].strip()
        else:
            text = raw_output.strip()
            
        return (time.time() - start, text)
    
    except Exception as e:
        duration = time.time() - start
        stop_model(model)
        return (duration, f"🚨 Error: {type(e).__name__}")

def run_test(image_path: str, include: str = None, exclude: str = None, 
            prompt: str = DEFAULT_PROMPT, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Execute transcription pipeline with custom parameters."""
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    image_b64 = encode_image(image_path)
    if not image_b64:
        return
    
    models = get_multimodal_models(include, exclude)
    if not models:
        print("❌ No matching multimodal models available")
        return
    
    print("=" * 80)
    print(f"Testing {len(models)} model{'s' if len(models) > 1 else ''}")
    print(f"Image: {image_path}")
    print(f"Timeout: {timeout}s | Include: {include or 'all'} | Exclude: {exclude or 'none'}")
    print("-" * 80)
    print("PROMPT:")
    print(prompt.strip())
    print("-" * 80)
    print(f"{'MODEL':<30} {'TIME':<8} {'RESULT'}")
    print("-" * 80)
    
    results = []
    for model in models:
        duration, text = force_transcription(model, image_b64, prompt, timeout)
        elapsed = f"{duration:.2f}s"
        preview = text.replace("\n", "\\n")[:60]  # Show newlines as escape sequences
        
        print(f"{model:<30} {elapsed:<8} {preview}")
        results.append((model, text))
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"transcription_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Test time: {timestamp}\n")
        f.write(f"Timeout: {timeout}s\n")
        f.write(f"Models: {len(models)} tested\n\n")
        f.write("PROMPT:\n")
        f.write(prompt.strip() + "\n\n")
        
        for model, text in results:
            f.write(f"\n{'='*50}\n# MODEL: {model}\n{'='*50}\n{text}\n")
    
    print("=" * 80)
    print(f"✅ Saved full results to: {output_file}")

def main():
    """Command-line interface with enhanced help documentation."""
    parser = argparse.ArgumentParser(
        description="Vision Model Transcription Benchmark",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "image_path",
        help="Path to image file for transcription"
    )
    parser.add_argument(
        "-i", "--include",
        help="Comma-separated patterns to include models (e.g. 'llama,gemma3')",
        metavar="PATTERNS"
    )
    parser.add_argument(
        "-e", "--exclude",
        help="Comma-separated patterns to exclude models (e.g. 'llava,7b')",
        metavar="PATTERNS"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT} = {DEFAULT_TIMEOUT//60} min)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom transcription prompt (use quotes for multi-line)"
    )
    
    # Extended help documentation
    default_prompt_preview = "\n".join([f"    {line}" for line in DEFAULT_PROMPT.strip().split("\n")])
    
    parser.epilog = f"""\
Examples:
  # Test all models with default settings
  python transcribe.py image.png
  
  # Custom timeout (5 minutes) and model filtering
  python transcribe.py document.jpg -t 300 -i llama3 -e 7b
  
  # Use custom prompt for specialized transcription
  python transcribe.py diagram.png -p "Extract all text EXACTLY as shown:"
  
  # Multi-line prompt (use quotes)
  python transcribe.py screenshot.png -p $'Line1\\nLine2\\nLine3'

Common model patterns:
  • Architecture: llama, gemma, mistral, fuyu
  • Modality: vision, llava, bakllava
  • Size: 7b, 13b, 70b
  • Version: v1, v2, v2.1, latest

Default Prompt:
{default_prompt_preview}

Note: Filters use case-insensitive substring matching"""

    # Handle help and no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    run_test(
        image_path=args.image_path,
        include=args.include,
        exclude=args.exclude,
        prompt=args.prompt,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()
