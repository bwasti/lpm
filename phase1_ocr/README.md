# Phase 1: OCR-like Text Normalization

Generate training data for teaching the Pixel Transformer to normalize text renderings.

## Overview

Creates input/output pairs where:
- **Input (X)**: Text rendered with variations (rotations, different fonts, blur, noise, variable sizes, alignments)
- **Output (Y)**: Same text rendered in clean, normalized style

This teaches the model OCR-like abilities to "clean up" text.

## Setup

```bash
cd phase1_ocr
```

## Files

- `generate_training_data.py` - Main script for generating training data
- `render.py` - Text rendering utilities with auto-sizing and alignment
- `visualize.py` - Interactive visualization tool (braille in terminal)

## Usage

### 1. Visualize Text Renderings

Test how text will be rendered with the interactive braille visualizer:

```bash
# Interactive mode - type text and see it rendered
python visualize.py

# Single render mode
python visualize.py "The quick brown fox"

# Try different variation levels
python visualize.py "Hello World" --variation high

# Validate with OCR to check text is recoverable
python visualize.py "Hello World" --variation medium --validate
```

**Interactive commands:**
- `:low` / `:medium` / `:high` - Change variation level
- `:quit` - Exit

### 2. Generate Training Data

Generate compressed batches of training data:

```bash
# Generate with local sample texts (for testing)
python generate_training_data.py --generate --local \
    --renderings 10 \
    --batch-size 1000 \
    --variation medium \
    --validate

# Generate from Wikipedia dataset (requires network)
python generate_training_data.py --generate \
    --max-articles 1000 \
    --renderings 10 \
    --batch-size 1000 \
    --variation medium \
    --validate
```

**Arguments:**
- `--generate` - Generate training data (required)
- `--local` - Use local sample texts instead of downloading
- `--dataset` - Dataset name (default: wikipedia)
- `--max-articles` - Number of articles to process
- `--renderings` - Number of variations per text (default: 10)
- `--batch-size` - Samples per batch file (default: 1000)
- `--variation` - Variation level: low/medium/high (default: medium)
- `--validate` - Validate with OCR (requires pytesseract + tesseract)
- `--output-dir` - Output directory (default: data/batches)

### 3. Output Format

**Compressed binary batches:**
```
data/batches/
├── batch_0000.gz  # ~600 bytes per sample
├── batch_0001.gz
└── ...
```

Each batch file contains:
- Binary bitmasks (1 bit per pixel, packed into bytes)
- Gzip compression for minimal disk usage
- Text metadata as JSON

**Loading batches:**
```python
from generate_training_data import DataBatch

# Load a batch
batch = DataBatch.load_compressed('data/batches/batch_0000.gz')

# Get sample
input_img, output_img, text = batch.get_sample(0)

# input_img, output_img are PIL Images (200x200 grayscale)
# text is the original string
```

## Rendering Variations

**Input variations:**
- Random fonts (serif, sans-serif, mono, bold)
- Random alignment (left, center, right)
- Auto-sizing to fit text in box (as large/small as possible)
- Rotation: -10° to +10° (medium)
- Gaussian blur: 0 to 1.0 radius
- Gaussian noise: 0 to 0.05 level
- Text wrapping with proper overflow handling

**Variation levels:**
- `low`: ±5° rotation, minimal blur/noise, ±15% size
- `medium`: ±10° rotation, moderate blur/noise, ±30% size
- `high`: ±20° rotation, heavy blur/noise, ±50% size

**Output (normalized):**
- Fixed clean font (DejaVu Sans)
- Center alignment
- Auto-sized to fit
- No rotation, blur, or noise

## OCR Validation

Use `--validate` to run Tesseract OCR on generated inputs:
- Samples with <30% accuracy are rejected
- Samples with >50% accuracy are marked as "passed"
- Pass rate typically 5-15% for medium variation
- Low pass rate is expected - the model learns to denoise!

**Requirements:**
```bash
pip install pytesseract
sudo dnf install tesseract  # Fedora/RHEL
# or: sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# or: brew install tesseract  # Mac
```

## Training

Load batches for training:

```python
from generate_training_data import DataBatch
from pathlib import Path
import numpy as np

# Load all batches
batch_files = sorted(Path('data/batches').glob('batch_*.gz'))
all_inputs = []
all_outputs = []
all_texts = []

for batch_file in batch_files:
    batch = DataBatch.load_compressed(batch_file)

    for i in range(len(batch)):
        input_img, output_img, text = batch.get_sample(i)

        # Convert to numpy arrays (0-1 range)
        input_arr = np.array(input_img).astype(np.float32) / 255.0
        output_arr = np.array(output_img).astype(np.float32) / 255.0

        all_inputs.append(input_arr)
        all_outputs.append(output_arr)
        all_texts.append(text)

# Stack into arrays
X_train = np.array(all_inputs)  # (N, 200, 200)
Y_train = np.array(all_outputs)  # (N, 200, 200)

# Train model (see ../train.py for reference)
```

## Storage Efficiency

- Raw PNG: ~5-10 KB per image
- Binary bitmask: ~5 KB per image (200×200÷8 bytes)
- Gzipped bitmask: ~600 bytes per image (~90% compression!)

For 10,000 samples:
- Raw PNGs: ~100 MB
- This format: ~6 MB

## Notes

- Same text appears in both input and output of each pair
- Input has random variations, output is always clean
- Model learns to normalize text appearance
- Auto-sizing ensures text fits and uses maximum available space
- Variable alignment teaches the model to handle different layouts
