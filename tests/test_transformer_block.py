"""
Test full transformer block
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
from gt.client import nn
import numpy as np
from model_clean import TransformerBlock

print("Testing full transformer block...")
print("=" * 60)

try:
    batch_size = 2
    seq_len = 4
    embed_dim = 128

    print(f"\nCreating transformer block...")
    block = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=256)
    print(f"  Block created with {len(list(block.parameters()))} parameters")

    print(f"\nCreating input...")
    x = gt.randn(batch_size, seq_len, embed_dim)
    print(f"  Input shape: {x.shape}")

    print(f"\nForward pass...")
    output = block(x)
    print(f"  Output shape: {output.shape}")

    loss = (output ** 2).mean()
    print(f"  Loss computed (not fetching value yet)")

    print(f"\nBackward pass...")
    loss.backward()
    print(f"  Backward: OK ✓")

    # Now try to fetch loss value
    print(f"\nFetching loss value...")
    try:
        loss_val = loss.data.numpy()
        print(f"  Loss: {loss_val:.4f} ✓")
    except Exception as e2:
        print(f"  Failed to fetch loss: {e2}")

    # Check gradients
    num_grads = sum(1 for p in block.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {num_grads}")

except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
