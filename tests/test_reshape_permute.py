"""
Test reshape and permute in attention-like pattern
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
import numpy as np

print("Testing reshape and permute operations...")
print("=" * 60)

try:
    batch_size = 2
    seq_len = 4
    embed_dim = 256
    num_heads = 4
    head_dim = embed_dim // num_heads

    print(f"\nTest parameters:")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")

    # Create input
    Q = gt.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    print(f"\nQ shape: {Q.shape}")

    # Reshape for multi-head
    print(f"\nReshaping to (batch, seq, heads, head_dim)...")
    Q_reshaped = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    print(f"  Q_reshaped shape: {Q_reshaped.shape}")

    # Permute to (batch, heads, seq, head_dim)
    print(f"\nPermuting to (batch, heads, seq, head_dim)...")
    Q_perm = Q_reshaped.permute(0, 2, 1, 3)
    print(f"  Q_perm shape: {Q_perm.shape}")

    # Compute loss
    loss = (Q_perm ** 2).mean()
    print(f"\nForward: OK, loss = {loss.data.numpy():.4f}")

    # Backward
    print(f"\nTesting backward...")
    loss.backward()
    print(f"Backward: OK ✓")

    print(f"\nGradient shape: {Q.grad.shape}")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
