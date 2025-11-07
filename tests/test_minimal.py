"""
Minimal test to isolate the backward pass bug
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
from gt.client import nn
import numpy as np

print("Testing backward pass with different components...")
print("=" * 60)

# Test 1: Simple linear layer
print("\n1. Testing simple linear layer...")
try:
    fc = nn.Linear(10, 5)
    x = gt.randn(2, 10)
    y = fc(x)
    loss = (y ** 2).mean()
    print(f"   Forward: OK, loss shape = {loss.shape}")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   Backward: FAILED - {e}")

# Test 2: Linear + ReLU
print("\n2. Testing linear + relu...")
try:
    fc = nn.Linear(10, 5)
    x = gt.randn(2, 10)
    y = fc(x).relu()
    loss = (y ** 2).mean()
    print(f"   Forward: OK, loss shape = {loss.shape}")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   Backward: FAILED - {e}")

# Test 3: With layer norm-like operations
print("\n3. Testing layernorm-like operations...")
try:
    x = gt.randn(2, 10)
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    eps = gt.from_numpy(np.array(1e-5, dtype='float32'))
    std = (variance + eps).sqrt()
    x_norm = (x - mean) / std

    # Scale and shift
    gamma = gt.ones(10, requires_grad=True)
    beta = gt.zeros(10, requires_grad=True)
    output = x_norm * gamma + beta

    loss = (output ** 2).mean()
    print(f"   Forward: OK, loss shape = {loss.shape}")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   Backward: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 4: Matmul operations (attention-like)
print("\n4. Testing matmul operations...")
try:
    Q = gt.randn(2, 4, 8)  # batch, seq, dim
    K = gt.randn(2, 4, 8)
    V = gt.randn(2, 4, 8)

    scores = Q @ K.T
    # Simple attention without softmax
    output = scores @ V
    loss = (output ** 2).mean()
    print(f"   Forward: OK, loss shape = {loss.shape}")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   Backward: FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
