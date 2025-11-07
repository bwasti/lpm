"""
Test LayerNorm operations step by step
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
import numpy as np

print("Testing LayerNorm operations step by step...")
print("=" * 60)

dim = 10
eps = 1e-5

# Create parameters
gamma = gt.ones(dim, requires_grad=True)
beta = gt.zeros(dim, requires_grad=True)

# Test with 3D input (like in transformer)
x = gt.randn(2, 4, dim)  # batch, seq, dim

print("\n1. Computing mean...")
mean = x.mean(axis=-1, keepdims=True)
print("   OK")

print("\n2. Computing variance...")
variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
print("   OK")

print("\n3. Computing std...")
eps_tensor = gt.from_numpy(np.array(eps, dtype='float32'))
std = (variance + eps_tensor).sqrt()
print("   OK")

print("\n4. Normalizing...")
x_norm = (x - mean) / std
print("   OK")

print("\n5. Scale and shift...")
output = x_norm * gamma + beta
print("   OK")

print("\n6. Computing loss...")
loss = (output ** 2).mean()
print("   OK")

print("\n7. Running backward...")
try:
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
