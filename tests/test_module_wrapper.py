"""
Test if Module wrapping causes the issue
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
from gt.client import nn
import numpy as np

print("Testing Module wrapping...")
print("=" * 60)

# Test 1: LayerNorm as a Module (copy of our implementation)
class SimpleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = gt.ones(dim, requires_grad=True)
        self.beta = gt.zeros(dim, requires_grad=True)
        self._parameters.extend([self.gamma, self.beta])

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        eps_tensor = gt.from_numpy(np.array(self.eps, dtype='float32'))
        std = (variance + eps_tensor).sqrt()
        x_norm = (x - mean) / std
        return x_norm * self.gamma + self.beta

print("\n1. Testing SimpleLayerNorm module...")
try:
    ln = SimpleLayerNorm(10)
    x = gt.randn(2, 4, 10)
    output = ln(x)
    loss = (output ** 2).mean()
    print("   Forward: OK")
    loss.backward()
    print("   Backward: OK ✓")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
