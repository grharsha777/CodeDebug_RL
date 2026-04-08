#!/usr/bin/env python3
"""Pre-submission checklist verification."""

import re
import sys
from pathlib import Path

def verify_inference_structure():
    """Verify inference.py follows sample structure."""
    inference_path = Path("inference.py")
    content = inference_path.read_text()
    
    checks = {
        "imports_openai": "from openai import OpenAI" in content,
        "has_load_local_env": "def load_local_env()" in content,
        "has_log_functions": all(
            f"def log_{name}(" in content 
            for name in ["start", "step", "end"]
        ),
        "has_run_task": "def run_task(" in content,
        "has_main": 'if __name__ == "__main__"' in content,
    }
    
    return all(checks.values()), checks

def verify_env_variables():
    """Verify environment variable handling."""
    inference_path = Path("inference.py")
    content = inference_path.read_text()
    
    # Extract variable definitions
    api_base = re.search(r'API_BASE_URL\s*=\s*os\.getenv\("API_BASE_URL",\s*"([^"]+)"\)', content)
    model = re.search(r'MODEL_NAME\s*=\s*os\.getenv\("MODEL_NAME",\s*"([^"]+)"\)', content)
    hf_token = re.search(r'HF_TOKEN\s*=\s*os\.getenv\("HF_TOKEN"\)', content)
    
    checks = {
        "API_BASE_URL_has_default": bool(api_base),
        "MODEL_NAME_has_default": bool(model),
        "HF_TOKEN_no_default": bool(hf_token),
        "HF_TOKEN_fallback_to_OPENAI_API_KEY": "os.getenv(\"OPENAI_API_KEY\")" in content,
        "HF_TOKEN_required": "if HF_TOKEN is None" in content,
    }
    
    return all(checks.values()), checks

def verify_openai_usage():
    """Verify OpenAI client is properly configured."""
    inference_path = Path("inference.py")
    content = inference_path.read_text()
    
    checks = {
        "client_initialized": "client = OpenAI(" in content,
        "base_url_set": "base_url=API_BASE_URL" in content,
        "api_key_set": "api_key=HF_TOKEN" in content,
        "chat_completions_used": "client.chat.completions.create(" in content,
        "model_param_used": "model=MODEL_NAME" in content,
    }
    
    return all(checks.values()), checks

def verify_structured_logging():
    """Verify structured logging format [START]/[STEP]/[END]."""
    inference_path = Path("inference.py")
    content = inference_path.read_text()
    
    checks = {
        "start_format": '[START]' in content and 'task=' in content,
        "step_format": '[STEP]' in content and 'step=' in content and 'action=' in content and 'reward=' in content,
        "end_format": '[END]' in content and 'success=' in content and 'steps=' in content,
        "log_functions_return_str": "-> str:" in content and 'return line' in content,
    }
    
    return all(checks.values()), checks

def verify_env_example():
    """.env.example should not have HF_TOKEN default."""
    env_path = Path(".env.example")
    if not env_path.exists():
        return False, {"file_exists": False}
    
    content = env_path.read_text()
    checks = {
        "file_exists": True,
        "has_API_BASE_URL": "API_BASE_URL" in content,
        "has_MODEL_NAME": "MODEL_NAME" in content,
        "has_HF_TOKEN": "HF_TOKEN" in content,
    }
    
    return all(checks.values()), checks

def main():
    """Run all verifications."""
    print("=" * 70)
    print("PRE-SUBMISSION CHECKLIST VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check 1: inference.py structure
    print("✓ Condition 1: inference.py follows sample structure")
    status, details = verify_inference_structure()
    for key, value in details.items():
        symbol = "✓" if value else "✗"
        print(f"    {symbol} {key.replace('_', ' ').title()}")
    all_passed = all_passed and status
    print()
    
    # Check 2: Environment variables
    print("✓ Condition 2: Environment variables are present")
    status, details = verify_env_variables()
    for key, value in details.items():
        symbol = "✓" if value else "✗"
        print(f"    {symbol} {key.replace('_', ' ').title()}")
    all_passed = all_passed and status
    print()
    
    # Check 3: Defaults only for API_BASE_URL and MODEL_NAME
    print("✓ Condition 3: Defaults only for API_BASE_URL and MODEL_NAME")
    status, _ = verify_env_variables()
    print(f"    {'✓' if status else '✗'} HF_TOKEN has no default, requires env var")
    all_passed = all_passed and status
    print()
    
    # Check 4: OpenAI client usage
    print("✓ Condition 4: All LLM calls use OpenAI client")
    status, details = verify_openai_usage()
    for key, value in details.items():
        symbol = "✓" if value else "✗"
        print(f"    {symbol} {key.replace('_', ' ').title()}")
    all_passed = all_passed and status
    print()
    
    # Check 5: Structured logging
    print("✓ Condition 5: Stdout logs follow structured format (START/STEP/END)")
    status, details = verify_structured_logging()
    for key, value in details.items():
        symbol = "✓" if value else "✗"
        print(f"    {symbol} {key.replace('_', ' ').title()}")
    all_passed = all_passed and status
    print()
    
    # Check 6: .env.example
    print("✓ Bonus: .env.example configuration")
    status, details = verify_env_example()
    for key, value in details.items():
        symbol = "✓" if value else "✗"
        print(f"    {symbol} {key.replace('_', ' ').title()}")
    print()
    
    print("=" * 70)
    if all_passed:
        print("STATUS: ALL PRE-SUBMISSION CONDITIONS SATISFIED ✓")
        print("=" * 70)
        return 0
    else:
        print("STATUS: SOME CONDITIONS NOT SATISFIED ✗")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
