"""
Training script for Pixel Transformer.

Demonstrates how to train on synthetic data.

Requires: gt.no_grad() for parameter updates
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'  # Enable auto-sharding for multi-GPU
os.environ['GT_VERBOSE'] = '1'     # Show what's happening

import gt
from gt.client import nn
import numpy as np
from model import PixelTransformer


def generate_synthetic_data(n_samples=100, seq_len=4, img_size=200, num_classes=10):
    """
    Generate synthetic pixel data for testing.

    Returns:
        X: (n_samples, seq_len, img_size, img_size) images
        y: (n_samples,) class labels
    """
    print(f"\nGenerating {n_samples} synthetic samples...")
    np.random.seed(42)

    # Random images
    X = np.random.randn(n_samples, seq_len, img_size, img_size).astype(np.float32) * 0.1

    # Random labels
    y = np.random.randint(0, num_classes, size=n_samples).astype(np.int64)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    return X, y


def train(model, X, y, epochs=10, lr=0.001, batch_size=2):
    """
    Train the model.

    Args:
        model: PixelTransformer model
        X: Training images (n_samples, seq_len, img_size, img_size)
        y: Training labels (n_samples,)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
    """
    n_samples = X.shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Convert to gt tensors
            X_tensor = gt.from_numpy(X_batch)
            y_tensor = gt.from_numpy(y_batch.astype('float32'))

            # Forward pass
            logits = model(X_tensor)

            # Simple loss: MSE with one-hot encoding
            # Convert labels to one-hot
            num_classes = logits.shape[-1]
            y_one_hot = np.zeros((len(y_batch), num_classes), dtype='float32')
            y_one_hot[np.arange(len(y_batch)), y_batch] = 1.0
            y_one_hot_tensor = gt.from_numpy(y_one_hot)

            # MSE loss
            loss = nn.mse_loss(logits, y_one_hot_tensor)

            # Backward pass
            loss.backward()

            # Parameter updates
            with gt.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                        param.grad.zero_()

            # Track metrics
            loss_value = loss.data.numpy()
            epoch_loss += loss_value

            # Compute accuracy
            logits_np = logits.data.numpy()
            predictions = np.argmax(logits_np, axis=1)
            correct += np.sum(predictions == y_batch)
            total += len(y_batch)

        # Print epoch statistics
        avg_loss = epoch_loss / num_batches
        accuracy = correct / total
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.4f}, Acc = {accuracy:.2%}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("Pixel Transformer Training")
    print("=" * 60)

    # Configuration
    img_size = 200
    seq_len = 4
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 256
    num_classes = 10
    max_seq_len = 16

    # Create model
    print("\nCreating model...")
    model = PixelTransformer(
        img_size=img_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        max_seq_len=max_seq_len
    )

    num_params = len(list(model.parameters()))
    print(f"  Model created with {num_params} parameter tensors")

    # Generate data
    X_train, y_train = generate_synthetic_data(
        n_samples=20,
        seq_len=seq_len,
        img_size=img_size,
        num_classes=num_classes
    )

    # Train
    train(
        model,
        X_train,
        y_train,
        epochs=5,
        lr=0.01,
        batch_size=2
    )


if __name__ == "__main__":
    main()
