#!/usr/bin/env python3
"""Test simple pour diagnostiquer le problème d'import qwen-vl-utils"""

import sys

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print("-" * 60)

# Test 1: import direct
print("\nTest 1: from qwen_vl_utils import process_vision_info")
try:
    from qwen_vl_utils import process_vision_info
    print("✅ SUCCESS")
    print(f"   process_vision_info: {process_vision_info}")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    print(f"   Type: {type(e).__name__}")

# Test 2: import module
print("\nTest 2: import qwen_vl_utils")
try:
    import qwen_vl_utils
    print("✅ SUCCESS")
    print(f"   Module: {qwen_vl_utils}")
    print(f"   __file__: {getattr(qwen_vl_utils, '__file__', 'N/A')}")
    if hasattr(qwen_vl_utils, "process_vision_info"):
        print(f"   process_vision_info: {qwen_vl_utils.process_vision_info}")
    else:
        print("   ⚠️  process_vision_info not found in module")
        print(f"   Available: {[x for x in dir(qwen_vl_utils) if not x.startswith('_')]}")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    print(f"   Type: {type(e).__name__}")

# Test 3: vérifier installation pip
print("\nTest 3: pip show qwen-vl-utils")
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pip", "show", "qwen-vl-utils"],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print("✅ Package trouvé:")
    for line in result.stdout.split("\n")[:10]:
        if line.strip():
            print(f"   {line}")
else:
    print("❌ Package non trouvé par pip")

print("\n" + "-" * 60)
