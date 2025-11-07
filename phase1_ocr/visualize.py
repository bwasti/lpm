"""
Interactive data inspector for Phase 1 OCR data.

Visualizes text renderings in the terminal using Unicode braille characters.
"""

import numpy as np
from PIL import Image
import argparse
from render import render_normal, render_varied

# Check if pytesseract is available
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# Unicode braille character mapping (2x4 pixel blocks)
# Braille pattern dots:
#   1 4
#   2 5
#   3 6
#   7 8
BRAILLE_START = 0x2800


def run_ocr(img):
    """Run OCR on a PIL image."""
    if not TESSERACT_AVAILABLE:
        return None

    # Convert to RGB if grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Run OCR
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return None


def image_to_braille(img_array, threshold=0.5):
    """
    Convert image to Unicode braille characters.

    Args:
        img_array: 2D numpy array (0-1 range, 0=black, 1=white)
        threshold: Threshold for converting to binary

    Returns:
        String with braille characters
    """
    height, width = img_array.shape

    # Braille characters represent 2x4 blocks
    braille_width = (width + 1) // 2
    braille_height = (height + 3) // 4

    # Pad to make dimensions divisible
    pad_height = braille_height * 4
    pad_width = braille_width * 2

    padded = np.ones((pad_height, pad_width))
    padded[:height, :width] = img_array

    # Convert to binary
    binary = (padded < threshold).astype(int)

    lines = []
    for row in range(braille_height):
        line = []
        for col in range(braille_width):
            # Get 2x4 block
            r = row * 4
            c = col * 2

            # Map pixels to braille dots
            # Dot pattern: 1=0x01, 2=0x02, 3=0x04, 4=0x08, 5=0x10, 6=0x20, 7=0x40, 8=0x80
            dots = 0
            if binary[r, c]:     dots |= 0x01      # dot 1
            if binary[r+1, c]:   dots |= 0x02      # dot 2
            if binary[r+2, c]:   dots |= 0x04      # dot 3
            if binary[r+3, c]:   dots |= 0x40      # dot 7
            if binary[r, c+1]:   dots |= 0x08      # dot 4
            if binary[r+1, c+1]: dots |= 0x10      # dot 5
            if binary[r+2, c+1]: dots |= 0x20      # dot 6
            if binary[r+3, c+1]: dots |= 0x80      # dot 8

            line.append(chr(BRAILLE_START + dots))

        lines.append(''.join(line))

    return '\n'.join(lines)


def display_images(img_input, img_output, text, ocr_text=None):
    """Display input and output images side by side with braille."""
    # Convert PIL images to numpy arrays
    input_array = np.array(img_input).astype(np.float32) / 255.0
    output_array = np.array(img_output).astype(np.float32) / 255.0

    # Convert to braille
    braille_input = image_to_braille(input_array, threshold=0.5)
    braille_output = image_to_braille(output_array, threshold=0.5)

    # Split into lines for side-by-side display
    input_lines = braille_input.split('\n')
    output_lines = braille_output.split('\n')

    # Calculate widths
    input_width = len(input_lines[0]) if input_lines else 0
    output_width = len(output_lines[0]) if output_lines else 0

    # Print header
    print("\n" + "=" * 80)
    print(f"Text: {text}")
    if ocr_text is not None:
        print(f"OCR:  {ocr_text}")
        # Calculate accuracy
        if ocr_text and text:
            matches = sum(1 for a, b in zip(ocr_text.lower(), text.lower()) if a == b)
            total = max(len(ocr_text), len(text))
            accuracy = matches / total * 100 if total > 0 else 0
            print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 80)

    # Print column headers
    header_input = "INPUT (Varied)".center(input_width)
    header_output = "OUTPUT (Clean)".center(output_width)
    print(f"{header_input}  │  {header_output}")
    print("─" * input_width + "  │  " + "─" * output_width)

    # Print images side by side
    for line_in, line_out in zip(input_lines, output_lines):
        print(f"{line_in}  │  {line_out}")

    print("=" * 80)


def interactive_mode():
    """Interactive mode - prompt for text and display."""
    print("\n" + "=" * 80)
    print("Phase 1 OCR Data Inspector")
    print("=" * 80)
    print("\nType text to see how it will be rendered for training.")
    print("Press Ctrl+C to exit.\n")

    variation_levels = ['low', 'medium', 'high']
    current_variation = 'medium'

    print(f"Current variation level: {current_variation}")
    print("Commands:")
    print("  :low    - Set variation to low")
    print("  :medium - Set variation to medium")
    print("  :high   - Set variation to high")
    print("  :quit   - Exit")
    print()

    try:
        while True:
            text = input("Enter text (or command): ").strip()

            if not text:
                continue

            # Handle commands
            if text.startswith(':'):
                cmd = text[1:].lower()
                if cmd in variation_levels:
                    current_variation = cmd
                    print(f"✓ Variation level set to: {current_variation}\n")
                    continue
                elif cmd == 'quit':
                    print("Goodbye!")
                    break
                else:
                    print(f"Unknown command: {text}\n")
                    continue

            # Render text
            try:
                print("\nRendering...")
                img_input = render_varied(text, size=200, font_size=20,
                                         variation_level=current_variation)
                img_output = render_normal(text, size=200, font_size=20)

                display_images(img_input, img_output, text)

            except Exception as e:
                print(f"Error rendering: {e}\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def single_render(text, variation='medium', save_prefix=None, validate=False):
    """Render a single text sample."""
    print(f"\nRendering: {text}")
    print(f"Variation: {variation}\n")

    img_input = render_varied(text, size=200, font_size=20,
                             variation_level=variation)
    img_output = render_normal(text, size=200, font_size=20)

    # Run OCR if requested
    ocr_text = None
    if validate:
        if not TESSERACT_AVAILABLE:
            print("Warning: pytesseract not available. Install with: pip install pytesseract")
            print("You also need the tesseract-ocr system package.\n")
        else:
            print("Running OCR...")
            ocr_text = run_ocr(img_input)
            if ocr_text is None:
                print("Warning: Tesseract binary not found. Install with:")
                print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
                print("  Mac: brew install tesseract\n")

    display_images(img_input, img_output, text, ocr_text=ocr_text)

    # Save if requested
    if save_prefix:
        img_input.save(f"{save_prefix}_input.png")
        img_output.save(f"{save_prefix}_output.png")
        print(f"\n✓ Saved to {save_prefix}_input.png and {save_prefix}_output.png")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect OCR training data with braille visualization'
    )
    parser.add_argument('text', nargs='*',
                       help='Text to render (if not provided, enters interactive mode)')
    parser.add_argument('--variation', choices=['low', 'medium', 'high'],
                       default='medium', help='Variation level')
    parser.add_argument('--save', type=str,
                       help='Save images with this prefix')
    parser.add_argument('--validate', action='store_true',
                       help='Run OCR validation to check text readability')

    args = parser.parse_args()

    if args.text:
        # Single render mode
        text = ' '.join(args.text)
        single_render(text, variation=args.variation, save_prefix=args.save,
                     validate=args.validate)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
