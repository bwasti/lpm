"""
Test multi-head attention separately
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
import numpy as np
from model_clean import MultiHeadAttention, LayerNorm

print("Testing components separately...")
print("=" * 60)

batch_size = 2
seq_len = 4
embed_dim = 128

# Test 1: LayerNorm
print("\n1. Testing LayerNorm...")
try:
    ln = LayerNorm(embed_dim)
    x = gt.randn(batch_size, seq_len, embed_dim)
    output = ln(x)
    loss = (output ** 2).mean()
    loss.backward()
    print("   LayerNorm: OK ✓")
except Exception as e:
    print(f"   LayerNorm: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 2: MultiHeadAttention
print("\n2. Testing MultiHeadAttention...")
try:
    mha = MultiHeadAttention(embed_dim, num_heads=4)
    x = gt.randn(batch_size, seq_len, embed_dim)
    output = mha(x)
    loss = (output ** 2).mean()
    print(f"   Forward: OK")
    loss.backward()
    print("   MultiHeadAttention: OK ✓")
except Exception as e:
    print(f"   MultiHeadAttention: FAILED - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
