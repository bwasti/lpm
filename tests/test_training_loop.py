"""
Test backward pass with tensors created from numpy in a loop (like training)
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'

import gt
from gt.client import nn
import numpy as np

print("Testing backward with numpy tensors in loop...")
print("=" * 60)

# Create a simple model
model = nn.Linear(10, 5)

# Test loop
for i in range(2):
    print(f"\nIteration {i+1}:")

    # Create tensors from numpy (like in training loop)
    X_np = np.random.randn(2, 10).astype('float32')
    y_np = np.random.randn(2, 5).astype('float32')

    X = gt.from_numpy(X_np)
    y = gt.from_numpy(y_np)

    # Forward
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    print(f"  Forward: OK, loss = {loss.data.numpy():.4f}")

    # Backward
    try:
        loss.backward()
        print(f"  Backward: OK ✓")

        # Update (like SGD)
        with gt.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= 0.01 * param.grad
                    param.grad.zero_()
        print(f"  Update: OK ✓")

    except Exception as e:
        print(f"  Backward: FAILED - {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "=" * 60)
