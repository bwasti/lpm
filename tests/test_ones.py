"""
Test if gt.ones() parameters cause backward issues
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
import numpy as np

print("Testing gt.ones() in backward pass...")
print("=" * 60)

# Test 1: Using gt.ones() as a parameter
print("\n1. Testing with gt.ones() parameter...")
try:
    x = gt.randn(2, 10)
    gamma = gt.ones(10, requires_grad=True)

    output = x * gamma
    loss = (output ** 2).mean()
    print("   Forward: OK")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 2: Using gt.from_numpy(np.ones()) as a parameter
print("\n2. Testing with gt.from_numpy(np.ones()) parameter...")
try:
    x = gt.randn(2, 10)
    gamma = gt.from_numpy(np.ones(10, dtype='float32'), requires_grad=True)

    output = x * gamma
    loss = (output ** 2).mean()
    print("   Forward: OK")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 3: Using gt.zeros() as a parameter
print("\n3. Testing with gt.zeros() parameter...")
try:
    x = gt.randn(2, 10)
    beta = gt.zeros(10, requires_grad=True)

    output = x + beta
    loss = (output ** 2).mean()
    print("   Forward: OK")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 4: Using randn as parameter (control)
print("\n4. Testing with gt.randn() parameter (control)...")
try:
    x = gt.randn(2, 10)
    w = gt.randn(10, requires_grad=True)

    output = x * w
    loss = (output ** 2).mean()
    print("   Forward: OK")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")

print("\n" + "=" * 60)
