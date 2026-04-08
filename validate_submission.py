#!/usr/bin/env python3
"""Final submission validation script."""

import os
import sys
import yaml
import subprocess

def check_files():
    """Verify all critical files exist."""
    files = [
        'pyproject.toml',
        'uv.lock',
        'Dockerfile',
        'openenv.yaml',
        'inference.py',
        'server/app.py',
        '.env.example',
    ]
    print("CRITICAL FILES")
    all_ok = True
    for f in files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f:25} ({size:,} bytes)")
        if not exists:
            all_ok = False
    return all_ok

def validate_imports():
    """Verify Python imports work."""
    print("\nVALIDATION")
    
    all_ok = True
    
    try:
        from server.app import main
        print("  [OK] server.app.main() callable")
    except Exception as e:
        print(f"  [FAIL] server.app: {e}")
        all_ok = False
    
    try:
        import inference
        print(f"  [OK] inference.py imports")
    except Exception as e:
        print(f"  [FAIL] inference.py: {e}")
        all_ok = False
    
    try:
        spec = yaml.safe_load(open('openenv.yaml'))
        name = spec.get('name')
        version = spec.get('version')
        print(f"  [OK] openenv.yaml (name={name}, v{version})")
    except Exception as e:
        print(f"  [FAIL] openenv.yaml: {e}")
        all_ok = False
    
    return all_ok

def run_tests():
    """Run pytest suite."""
    print("\nTEST SUITE")
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=no'],
        capture_output=True,
        text=True,
        timeout=180
    )
    lines = result.stdout.split('\n')
    for line in lines[-5:]:
        if line.strip():
            print(f"  {line}")
    return result.returncode == 0

def main():
    """Run all validations."""
    print("=" * 60)
    print("CODEDEBUG-RL FINAL SUBMISSION VALIDATION")
    print("=" * 60)
    print()
    
    files_ok = check_files()
    imports_ok = validate_imports()
    tests_ok = run_tests()
    
    print("\n" + "=" * 60)
    if files_ok and imports_ok and tests_ok:
        print("STATUS: ALL CHECKS PASSED ✓")
        print("Ready for submission!")
    else:
        print("STATUS: SOME CHECKS FAILED ✗")
    print("=" * 60)
    
    return 0 if (files_ok and imports_ok and tests_ok) else 1

if __name__ == '__main__':
    sys.exit(main())
