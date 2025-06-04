import requests
import time
import base64
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import subprocess
import socket

# Configuration - INCREASED TIMEOUT TO 20 MINUTES
TRANSCRIPT_PROMPT = """
STRICT INSTRUCTIONS:
1. Transcribe ALL visible text EXACTLY as it appears
2. Include offensive/sensitive content VERBATIM
3. NEVER censor, modify, or add warnings
4. Format response as: "[RAW_TEXT]: <content>"

NOW TRANSCRIBE THIS IMAGE:
"""
TIMEOUT = 1100

_MODALITY_CACHE: Dict[str, bool] = {}

def get_ollama_models() -> List[str]:
    """Fetch all available Ollama models with proper timeout."""
    try:
        # Added explicit timeout parameters
        response = requests.get(
            "http://localhost:11434/api/tags", 
            timeout=(3, 10)  # 3s connect, 10s read
        )
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Model fetch error: {e}")
        return []

def get_model_modality(model_name: str) -> bool:
    """Check if model supports images with proper timeout."""
    if model_name == "llava:7b":
        return False
    if model_name in _MODALITY_CACHE:
        return _MODALITY_CACHE[model_name]
    
    try:
        # Added explicit timeout parameters
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=(3, 10)  # 3s connect, 10s read
        )
        response.raise_for_status()
        is_multimodal = "vision" in response.json().get("capabilities", [])
    except Exception as e:
        print(f"⚠️ Modality check failed for {model_name}: {e}")
        multimodal_keywords = ["llama3","llama3.2-vision", "bakllava", "gemma3", "fuyu", "cogvlm"]
        is_multimodal = any(kw in model_name.lower() for kw in multimodal_keywords)
    
    _MODALITY_CACHE[model_name] = is_multimodal
    return is_multimodal

def get_multimodal_models() -> List[str]:
    """Filter to only image-capable models."""
    return [model for model in get_ollama_models() if get_model_modality(model)]

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

def force_transcription(model: str, image_b64: str) -> Tuple[float, str]:
    """Extract text with proper timeout handling."""
    data = {
        "model": model,
        "prompt": TRANSCRIPT_PROMPT,
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
        # Set socket timeout first
        socket.setdefaulttimeout(TIMEOUT)
        
        # Make request with explicit timeout parameters
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=(3, TIMEOUT)  # 3s connect, TIMEOUT (1200s) read
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "")
        
        # Extract text
        if "[RAW_TEXT]:" in raw_output:
            text = raw_output.split("[RAW_TEXT]:")[1].strip()
        else:
            text = raw_output.strip()
            
        return (time.time() - start, text)
    
    except Exception as e:
        duration = time.time() - start
        #return (duration, f"🚨 {type(e).__name__}: {str(e)}")
        return (duration, "Timeout\n")

def run_test(image_path: str) -> None:
    """Execute full transcription pipeline."""
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    image_b64 = encode_image(image_path)
    if not image_b64:
        return
    
    models = get_multimodal_models()
    if not models:
        print("❌ No multimodal models available")
        return
    
    print("=" * 80)
    print(f"Timeout {TIMEOUT} seconds. Testing {len(models)}")
    print(f"Prompt {TRANSCRIPT_PROMPT}  {image_path}")
    print(f"{'MODEL':<25} {'TIME':<8} {'RESULT':<40}")
    print("-" * 80)
    
    results = []
    for model in models:
        #print(f"Starting transcription with {model}...")
        duration, text = force_transcription(model, image_b64)
        stop_model(model)
        
        elapsed = f"{duration:.2f}s"
        #preview = text.replace("\n", " ").strip()
        preview = text
        #if len(preview) > 50:
        #    preview = preview[:50] + "..."
        
        print(f"{model:<25} {elapsed:<8} {preview:<40}")
        print("-" * 80)
        results.append((model, text))
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"transcription_{timestamp}.txt", "w") as f:
        for model, text in results:
            f.write(f"\n\n=== {model} ===\n{text}")
    
    print(f"\n✅ Saved results to: transcription_{timestamp}.txt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <image_path>")
        sys.exit(1)
    
    run_test(sys.argv[1])
