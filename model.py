"""
Pixel Transformer - A transformer that operates on 200x200 pixel images as tokens.

Each "token" is a full 200x200 bitmap (40,000 dimensional vector).
"""

import gt
from gt.client import nn
import numpy as np


def softmax(x, axis=-1):
    """
    Softmax activation along specified axis.
    softmax(x) = exp(x) / sum(exp(x))

    Uses numerically stable version: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    # Numerically stable softmax
    max_val = x.max(axis=axis, keepdims=True)
    exp_x = (x - max_val).exp()
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes across the feature dimension:
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale and shift parameters
        self.gamma = gt.ones(dim, requires_grad=True)
        self.beta = gt.zeros(dim, requires_grad=True)
        self._parameters.extend([self.gamma, self.beta])

    def forward(self, x):
        # x shape: (batch, seq_len, dim) or (batch, dim)
        mean = x.mean(axis=-1, keepdims=True)
        # Variance: E[(x - mean)^2]
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        # Normalize
        eps_tensor = gt.from_numpy(np.array(self.eps, dtype='float32'))
        std = (variance + eps_tensor).sqrt()
        x_norm = (x - mean) / std

        # Scale and shift
        return x_norm * self.gamma + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Scale factor for scaled dot-product attention
        self.scale = gt.from_numpy(np.array(self.head_dim ** -0.5, dtype='float32'))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)  # (batch, seq_len, embed_dim)
        V = self.v_proj(x)  # (batch, seq_len, embed_dim)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        # Then transpose to: (batch, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # Q: (batch, num_heads, seq_len, head_dim)
        # K: (batch, num_heads, seq_len, head_dim)
        # Scores: (batch, num_heads, seq_len, seq_len)

        # K transpose for matmul: (batch, num_heads, head_dim, seq_len)
        K_T = K.transpose(2, 3)

        # Attention scores
        scores = (Q @ K_T) * self.scale  # (batch, num_heads, seq_len, seq_len)
        attn_weights = softmax(scores, axis=-1)

        # Apply attention to values
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        # V: (batch, num_heads, seq_len, head_dim)
        # output: (batch, num_heads, seq_len, head_dim)
        attn_output = attn_weights @ V

        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)

        # Reshape: (batch, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = ReLU(xW1 + b1)W2 + b2
    """
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.

    x -> LayerNorm -> MultiHeadAttention -> Add -> LayerNorm -> FFN -> Add
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x


class PixelTransformer(nn.Module):
    """
    Transformer that processes sequences of 200x200 pixel images.

    Each image (token) is:
    1. Flattened to 40,000 dims
    2. Projected to embed_dim
    3. Added positional encoding (optional)
    4. Passed through transformer layers
    5. Pooled and projected to output

    Args:
        img_size: Size of square images (default: 200)
        embed_dim: Embedding dimension for transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        ff_dim: Feed-forward hidden dimension
        num_classes: Number of output classes (for classification)
        max_seq_len: Maximum sequence length for positional encoding
    """
    def __init__(self, img_size=200, embed_dim=256, num_heads=4,
                 num_layers=4, ff_dim=1024, num_classes=10, max_seq_len=32):
        super().__init__()

        self.img_size = img_size
        self.pixel_dim = img_size * img_size  # 40,000 for 200x200
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Project flattened pixels to embedding dimension
        self.pixel_embedding = nn.Linear(self.pixel_dim, embed_dim)

        # Learnable positional encodings (stored as numpy to avoid sharding issues)
        # Shape: (max_seq_len, embed_dim)
        self.pos_encoding_data = np.random.randn(max_seq_len, embed_dim).astype('float32') * 0.01
        # We'll create the tensor on-demand in forward pass to avoid slicing sharded tensors

        # Transformer blocks
        self.blocks = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, ff_dim)
            self.blocks.append(block)
            setattr(self, f'block_{i}', block)  # Register as module

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, img_size, img_size) - sequence of images
               OR (batch, seq_len, img_size*img_size) - flattened images
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Flatten images if not already flattened
        if len(x.shape) == 4:
            # Reshape (batch, seq_len, img_size, img_size) -> (batch, seq_len, img_size*img_size)
            x = x.reshape(batch_size, seq_len, self.pixel_dim)

        # Project pixels to embeddings
        x = self.pixel_embedding(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding (optional)
        # Note: Currently using fixed positional encodings created on-demand
        # to avoid slicing sharded tensors
        with gt.no_grad():
            pos_data = np.zeros((batch_size, seq_len, self.embed_dim), dtype='float32')
            for i in range(seq_len):
                pos_data[:, i, :] = self.pos_encoding_data[i, :]
            pos = gt.from_numpy(pos_data)
        x = x + pos

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling: average over sequence dimension
        # x: (batch, seq_len, embed_dim) -> (batch, embed_dim)
        pooled = x.mean(axis=1)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits


# Example usage
if __name__ == "__main__":
    print("Pixel Transformer Model")
    print("=" * 60)

    try:
        # Model configuration
        print("\nCreating model...")
        model = PixelTransformer(
            img_size=200,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
            num_classes=10,
            max_seq_len=16
        )

        # Create dummy data: batch of 2, sequence of 4 images
        batch_size = 2
        seq_len = 4
        print(f"\nCreating dummy input: ({batch_size}, {seq_len}, 200, 200)")
        dummy_images = gt.randn(batch_size, seq_len, 200, 200)

        print(f"Input shape: {dummy_images.shape}")
        print(f"Model: {len(list(model.parameters()))} parameter tensors")

        # Forward pass
        print("\nRunning forward pass...")
        logits = model(dummy_images)

        print(f"Output shape: {logits.shape}")
        print(f"Expected: ({batch_size}, 10)")

        print("\n" + "=" * 60)
        print("✓ Model successfully created and tested!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        print("\nCheck MISSING.md for required features")
        import traceback
        traceback.print_exc()
