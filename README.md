# Large Pixel Model (LPM) - Pixel-based Transformer

A transformer architecture that operates on sequences of 200×200 pixel images as tokens, built using the `gt` distributed tensor framework.

## Overview

This model treats each entire 200×200 image as a single token (40,000 dimensions). This is different from traditional vision transformers which use small patches (e.g., 16×16).

## Quick Start

```bash
# Run training on synthetic data
python train.py

# Test the model
python model.py
```

## Project Structure

```
lpm/
├── model.py              # Pixel Transformer implementation
├── train.py              # Training script with synthetic data
├── tests/                # Test files for components
├── README.md             # This file
├── MISSING.md            # Feature status and bug documentation
└── TEST_RESULTS.md       # Testing results
```

## Architecture

### Model Pipeline

1. **Input**: Sequence of 200×200 pixel images `(batch, seq_len, 200, 200)`
2. **Pixel Embedding**: Linear projection from 40,000 dims to `embed_dim`
3. **Positional Encoding**: Fixed positional embeddings
4. **Transformer Layers**: Standard encoder blocks
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - Residual connections
5. **Output**: Global average pooling followed by classification head

### Configuration

```python
from model import PixelTransformer

model = PixelTransformer(
    img_size=200,        # Size of square input images
    embed_dim=256,       # Embedding dimension
    num_heads=4,         # Number of attention heads
    num_layers=4,        # Number of transformer blocks
    ff_dim=1024,         # Feed-forward hidden dimension
    num_classes=10,      # Number of output classes
    max_seq_len=32       # Maximum sequence length
)
```

## Usage

### Basic Forward Pass

```python
import os
os.environ['GT_AUTO_SHARD'] = '1'  # Enable multi-GPU

import gt
from model import PixelTransformer

model = PixelTransformer(
    img_size=200,
    embed_dim=256,
    num_heads=4,
    num_layers=4
)

images = gt.randn(batch_size, seq_len, 200, 200)
logits = model(images)  # (batch_size, num_classes)
```

### Training

```python
import os
os.environ['GT_AUTO_SHARD'] = '1'  # Auto multi-GPU
os.environ['GT_VERBOSE'] = '1'     # Show GPU info

# python train.py
```

See `train.py` for training loop implementation including synthetic data generation, gradient descent updates, and accuracy tracking.

## Performance

Training on synthetic data (8 GPUs):
```
Epoch  1/5: Loss = 0.1003, Acc = 10.00%
Epoch  2/5: Loss = 0.0992, Acc = 20.00%
Epoch  3/5: Loss = 0.0982, Acc = 30.00%
Epoch  4/5: Loss = 0.0972, Acc = 30.00%
Epoch  5/5: Loss = 0.0963, Acc = 40.00%
```

- Startup time: ~1.8 seconds (server + 8 workers)
- Training time: ~2 seconds for 5 epochs (20 samples)
- Model size: 36 parameter tensors

## Implementation Details

### Multi-Head Attention

Reshapes input to `(batch, heads, seq, head_dim)` for parallel attention computation. Uses scaled dot-product attention with softmax.

### LayerNorm

Computes mean and variance along feature dimension. Applies learnable scale and shift parameters. Numerical stability parameter eps=1e-5.

### Positional Encoding

Currently implemented as fixed encodings created on-demand during forward pass. This avoids tensor slicing issues with sharded tensors. Could be made learnable if slicing on sharded tensors is supported in the future.

## Testing

```bash
# Test individual components
python tests/test_components.py
python tests/test_transformer_block.py

# Test training loop
python tests/test_training_loop.py

# Run full training
python train.py
```

See `TEST_RESULTS.md` for detailed test results.

## Requirements

- Python 3.8+
- `gt` framework (https://github.com/bwasti/gt)
- NumPy

## License

MIT
