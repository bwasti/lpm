"""
Generate OCR training data from text corpus.

Downloads text data (Wikipedia), renders to 200x200 bitmaps with variations,
and saves as compressed binary batches.
"""

import numpy as np
import argparse
from pathlib import Path
import gzip
import struct
import json
from PIL import Image
from render import render_varied, render_normal


def get_sample_texts():
    """Generate sample texts for testing."""
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers to progressively extract higher-level features from raw input.",
        "Natural language processing enables computers to understand, interpret and generate human language.",
        "Computer vision is an interdisciplinary field that deals with how computers can gain understanding from images.",
        "The Transformer architecture revolutionized natural language processing with its attention mechanism.",
        "Artificial intelligence has applications in robotics, healthcare, finance, and many other domains.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data.",
    ]
    return samples * 10  # Repeat to get more samples


def load_dataset(dataset_name='wikipedia', language='en', max_samples=None, use_local=False):
    """
    Load text dataset.

    Args:
        dataset_name: Name of dataset (default: 'wikipedia')
        language: Language code (default: 'en')
        max_samples: Maximum number of samples to load (None = all)
        use_local: Use local sample data instead of downloading (default: False)

    Returns:
        List of text strings
    """
    print(f"\nLoading dataset: {dataset_name} ({language})")

    # Option 1: Use sample texts for testing
    if use_local:
        print("Using local sample texts...")
        texts = get_sample_texts()
        if max_samples:
            texts = texts[:max_samples]
        print(f"✓ Loaded {len(texts)} sample texts")
        return texts

    # Option 2: Try to load from HuggingFace
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        print("\nFalling back to sample texts...")
        return get_sample_texts()

    # Load Wikipedia dataset
    try:
        print("Downloading/loading from HuggingFace...")
        dataset = hf_load_dataset(
            dataset_name,
            language=language,
            split='train',
            streaming=True  # Stream to avoid loading all at once
        )

        # Extract text samples
        texts = []
        print(f"Extracting text samples...")

        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            # Wikipedia items have 'text' field
            text = item.get('text', '').strip()

            if text:
                # Split into sentences/paragraphs for variety
                # Keep paragraphs that are reasonable length
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if 10 <= len(para) <= 500:  # Reasonable length for rendering
                        texts.append(para)

            # Progress
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1} articles, extracted {len(texts)} text samples")

        print(f"\n✓ Loaded {len(texts)} text samples")
        return texts

    except Exception as e:
        print(f"ERROR loading from HuggingFace: {e}")
        print("\nFalling back to sample texts...")
        return get_sample_texts()


def image_to_bitmask(img, threshold=128):
    """
    Convert PIL image to binary bitmask (1 bit per pixel).

    Args:
        img: PIL Image (grayscale)
        threshold: Threshold for binary conversion (0-255)

    Returns:
        numpy array of uint8 where each byte packs 8 pixels
    """
    # Convert to numpy array
    arr = np.array(img)

    # Binarize: 1 for black text, 0 for white background
    binary = (arr < threshold).astype(np.uint8)

    # Pack bits: 8 pixels per byte
    height, width = binary.shape
    assert height == width, "Image must be square"

    # Flatten and pack
    flat = binary.flatten()
    num_bytes = (len(flat) + 7) // 8  # Round up

    # Pad to multiple of 8
    if len(flat) % 8 != 0:
        flat = np.pad(flat, (0, 8 - len(flat) % 8), 'constant')

    # Pack 8 bits into each byte
    packed = np.packbits(flat)

    return packed


def bitmask_to_image(packed, size=200):
    """
    Convert packed binary bitmask back to PIL image.

    Args:
        packed: numpy array of uint8 (packed bits)
        size: Image size (square)

    Returns:
        PIL Image (grayscale)
    """
    # Unpack bits
    unpacked = np.unpackbits(packed)

    # Reshape and trim to size
    total_pixels = size * size
    binary = unpacked[:total_pixels].reshape(size, size)

    # Convert back: 1 (black) -> 0, 0 (white) -> 255
    arr = (1 - binary) * 255

    return Image.fromarray(arr.astype(np.uint8))


class DataBatch:
    """
    A batch of training samples with compressed binary storage.
    """

    def __init__(self, size=200):
        self.size = size
        self.inputs = []  # List of packed bitmasks
        self.outputs = []  # List of packed bitmasks
        self.texts = []  # List of text strings

    def add_sample(self, input_img, output_img, text):
        """Add a training sample to the batch."""
        # Convert images to packed bitmasks
        input_packed = image_to_bitmask(input_img)
        output_packed = image_to_bitmask(output_img)

        self.inputs.append(input_packed)
        self.outputs.append(output_packed)
        self.texts.append(text)

    def save_compressed(self, filepath):
        """
        Save batch as compressed binary file.

        File format:
        - Header: size (uint32), num_samples (uint32)
        - For each sample:
            - input bitmask (packed bytes)
            - output bitmask (packed bytes)
        - Texts as JSON (at end)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(filepath, 'wb') as f:
            # Write header
            num_samples = len(self.inputs)
            f.write(struct.pack('II', self.size, num_samples))

            # Write bitmasks
            for input_packed, output_packed in zip(self.inputs, self.outputs):
                # Write input
                f.write(input_packed.tobytes())
                # Write output
                f.write(output_packed.tobytes())

            # Write texts as JSON
            texts_json = json.dumps(self.texts).encode('utf-8')
            f.write(struct.pack('I', len(texts_json)))
            f.write(texts_json)

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def load_compressed(filepath):
        """Load batch from compressed binary file."""
        batch = DataBatch()

        with gzip.open(filepath, 'rb') as f:
            # Read header
            size, num_samples = struct.unpack('II', f.read(8))
            batch.size = size

            # Calculate bytes per image
            bytes_per_image = (size * size + 7) // 8

            # Read bitmasks
            for _ in range(num_samples):
                # Read input
                input_bytes = f.read(bytes_per_image)
                input_packed = np.frombuffer(input_bytes, dtype=np.uint8)
                batch.inputs.append(input_packed)

                # Read output
                output_bytes = f.read(bytes_per_image)
                output_packed = np.frombuffer(output_bytes, dtype=np.uint8)
                batch.outputs.append(output_packed)

            # Read texts
            texts_len = struct.unpack('I', f.read(4))[0]
            texts_json = f.read(texts_len).decode('utf-8')
            batch.texts = json.loads(texts_json)

        return batch

    def get_sample(self, idx):
        """Get a single sample as PIL images and text."""
        input_img = bitmask_to_image(self.inputs[idx], self.size)
        output_img = bitmask_to_image(self.outputs[idx], self.size)
        return input_img, output_img, self.texts[idx]


def validate_with_ocr(img, expected_text):
    """
    Validate rendered text with OCR.

    Returns:
        (success, ocr_text, accuracy) tuple
    """
    try:
        import pytesseract

        # Convert to RGB for tesseract
        if img.mode != 'RGB':
            img = img.convert('RGB')

        ocr_text = pytesseract.image_to_string(img).strip()

        # Calculate character-level accuracy
        if ocr_text and expected_text:
            # Normalize whitespace
            pred = ' '.join(ocr_text.lower().split())
            exp = ' '.join(expected_text.lower().split())

            matches = sum(1 for p, e in zip(pred, exp) if p == e)
            total = max(len(pred), len(exp))
            accuracy = matches / total if total > 0 else 0.0

            # Consider successful if >50% accurate
            success = accuracy > 0.5
            return success, ocr_text, accuracy
        else:
            return False, ocr_text, 0.0

    except ImportError:
        return None, None, None  # OCR not available
    except Exception as e:
        return False, str(e), 0.0


def generate_training_data(
    texts,
    output_dir='data/batches',
    batch_size=1000,
    renderings_per_text=10,
    variation_level='medium',
    validate_ocr=False,
    size=200
):
    """
    Generate training data from text samples.

    Args:
        texts: List of text strings
        output_dir: Directory to save batches
        batch_size: Number of samples per batch file
        renderings_per_text: Number of variations to render per text
        variation_level: Variation level for inputs ('low', 'medium', 'high')
        validate_ocr: Whether to validate with OCR
        size: Image size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Generating training data")
    print(f"{'='*80}")
    print(f"Texts: {len(texts)}")
    print(f"Renderings per text: {renderings_per_text}")
    print(f"Variation level: {variation_level}")
    print(f"Batch size: {batch_size}")
    print(f"Validate OCR: {validate_ocr}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    current_batch = DataBatch(size=size)
    batch_num = 0
    total_samples = 0
    total_validated = 0
    total_passed = 0

    for text_idx, text in enumerate(texts):
        # Generate multiple renderings for this text
        for render_idx in range(renderings_per_text):
            # Render with auto-sizing
            input_img = render_varied(text, size=size, auto_size=True, variation_level=variation_level)
            output_img = render_normal(text, size=size, auto_size=True)

            # Validate with OCR if requested
            if validate_ocr:
                success, ocr_text, accuracy = validate_with_ocr(input_img, text)

                if success is None:
                    # OCR not available, skip validation
                    validate_ocr = False
                    print("OCR validation not available (pytesseract not installed)")
                elif success:
                    total_validated += 1
                    total_passed += 1
                else:
                    total_validated += 1
                    # Skip samples that fail OCR validation
                    if accuracy < 0.3:  # Very poor quality, skip
                        continue

            # Add to batch
            current_batch.add_sample(input_img, output_img, text)
            total_samples += 1

            # Save batch if full
            if len(current_batch) >= batch_size:
                batch_file = output_dir / f"batch_{batch_num:04d}.gz"
                current_batch.save_compressed(batch_file)
                file_size = batch_file.stat().st_size

                print(f"✓ Saved {batch_file.name}: {len(current_batch)} samples, "
                      f"{file_size/1024:.1f} KB ({file_size/len(current_batch):.0f} bytes/sample)")

                # Start new batch
                current_batch = DataBatch(size=size)
                batch_num += 1

        # Progress
        if (text_idx + 1) % 10 == 0:
            print(f"  Processed {text_idx + 1}/{len(texts)} texts, "
                  f"generated {total_samples} samples...")

    # Save final batch if not empty
    if len(current_batch) > 0:
        batch_file = output_dir / f"batch_{batch_num:04d}.gz"
        current_batch.save_compressed(batch_file)
        file_size = batch_file.stat().st_size

        print(f"✓ Saved {batch_file.name}: {len(current_batch)} samples, "
              f"{file_size/1024:.1f} KB ({file_size/len(current_batch):.0f} bytes/sample)")

    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Total batches: {batch_num + 1}")
    if total_validated > 0:
        print(f"  OCR pass rate: {total_passed}/{total_validated} ({total_passed/total_validated*100:.1f}%)")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate OCR training data from text corpus'
    )
    parser.add_argument('--dataset', default='wikipedia',
                       help='Dataset name (default: wikipedia)')
    parser.add_argument('--language', default='en',
                       help='Language code (default: en)')
    parser.add_argument('--max-articles', type=int, default=100,
                       help='Maximum articles to process (default: 100)')
    parser.add_argument('--local', action='store_true',
                       help='Use local sample texts instead of downloading')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - just load and display samples')

    # Generation options
    parser.add_argument('--generate', action='store_true',
                       help='Generate training data')
    parser.add_argument('--output-dir', default='data/batches',
                       help='Output directory for batches (default: data/batches)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Samples per batch file (default: 1000)')
    parser.add_argument('--renderings', type=int, default=10,
                       help='Number of renderings per text (default: 10)')
    parser.add_argument('--variation', choices=['low', 'medium', 'high'],
                       default='medium', help='Variation level (default: medium)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate with OCR (requires pytesseract)')

    args = parser.parse_args()

    # Load dataset
    texts = load_dataset(
        dataset_name=args.dataset,
        language=args.language,
        max_samples=args.max_articles,
        use_local=args.local
    )

    if args.generate:
        # Generate training data
        generate_training_data(
            texts=texts,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            renderings_per_text=args.renderings,
            variation_level=args.variation,
            validate_ocr=args.validate
        )

    elif args.test:
        print("\n" + "=" * 80)
        print("TEST MODE - Sample texts:")
        print("=" * 80)
        for i, text in enumerate(texts[:5]):
            print(f"\nSample {i+1}:")
            print(f"  Length: {len(text)} chars")
            print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print("\n" + "=" * 80)
        print(f"Total samples available: {len(texts)}")
        print("\n" + "=" * 80)
        print("Testing binary batch format...")
        print("=" * 80)

        # Create a small test batch
        test_batch = DataBatch(size=200)

        for i, text in enumerate(texts[:3]):
            print(f"\nRendering sample {i+1}: {text[:50]}...")
            # Render with auto-sizing
            input_img = render_varied(text, size=200, auto_size=True, variation_level='medium')
            output_img = render_normal(text, size=200, auto_size=True)

            test_batch.add_sample(input_img, output_img, text)

        # Save batch
        test_file = Path('data/test_batch.gz')
        test_batch.save_compressed(test_file)
        file_size = test_file.stat().st_size
        print(f"\n✓ Saved test batch to {test_file}")
        print(f"  Samples: {len(test_batch)}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"  Bytes per sample: {file_size/len(test_batch):.0f}")

        # Test loading
        print("\nTesting load...")
        loaded_batch = DataBatch.load_compressed(test_file)
        print(f"✓ Loaded {len(loaded_batch)} samples")

        # Verify
        img1, img2, text = loaded_batch.get_sample(0)
        print(f"\nSample 0 text: {text[:50]}...")
        print(f"  Input image size: {img1.size}")
        print(f"  Output image size: {img2.size}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
