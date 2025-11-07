"""
Text rendering utilities for generating OCR training data.

Renders text to 200x200 pixel images with various styles and transformations.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os


def get_available_fonts():
    """Get list of available system fonts."""
    # Common font paths on Linux/Mac/Windows
    font_dirs = [
        "/usr/share/fonts/truetype",
        "/System/Library/Fonts",
        "C:\\Windows\\Fonts",
    ]

    fonts = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if file.endswith(('.ttf', '.otf')):
                        fonts.append(os.path.join(root, file))

    return fonts


def get_default_fonts():
    """Get a curated list of common fonts for variation."""
    # Try to find common readable fonts (including bold variants)
    font_names = [
        # Sans-serif
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "Arial.ttf",
        "Arial-Bold.ttf",
        "Helvetica.ttf",
        "Helvetica-Bold.ttf",
        # Serif
        "DejaVuSerif.ttf",
        "DejaVuSerif-Bold.ttf",
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "Times.ttf",
        "Times-Bold.ttf",
        "Georgia.ttf",
        "Georgia-Bold.ttf",
        # Mono
        "DejaVuSansMono.ttf",
        "DejaVuSansMono-Bold.ttf",
        "LiberationMono-Regular.ttf",
        "LiberationMono-Bold.ttf",
        "Courier.ttf",
        "Courier-Bold.ttf",
    ]

    available = get_available_fonts()
    found = []

    for font_name in font_names:
        for font_path in available:
            if font_name in font_path:
                found.append(font_path)
                break

    # Fallback to any available fonts if we didn't find the common ones
    if not found:
        found = available[:10] if len(available) >= 10 else available

    return found


def find_optimal_font_size(text, font_path, size, draw, min_size=8, max_size=100):
    """
    Find the largest font size that fits the text in the given image size.

    Args:
        text: Text to render
        font_path: Path to font file
        size: Image size (square)
        draw: ImageDraw object
        min_size: Minimum font size to try
        max_size: Maximum font size to try

    Returns:
        Optimal font size
    """
    margin = 20  # Margin from edges
    max_width = size - margin
    max_height = size - margin

    # Binary search for optimal font size
    best_size = min_size
    left, right = min_size, max_size

    while left <= right:
        mid = (left + right) // 2

        # Try this font size
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, mid)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            break

        # Wrap text at this size
        wrapped = wrap_text(text, font, max_width, draw)

        # Check if it fits
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= max_width and text_height <= max_height:
            # Fits! Try larger
            best_size = mid
            left = mid + 1
        else:
            # Too big, try smaller
            right = mid - 1

    return best_size


def render_text(text, size=200, font_path=None, font_size=20,
                rotation=0, blur=0, noise=0, align='center', auto_size=False):
    """
    Render text to a square image.

    Args:
        text: Text to render
        size: Image size (square)
        font_path: Path to TTF font file (None for default)
        font_size: Font size in points (ignored if auto_size=True)
        rotation: Rotation angle in degrees
        blur: Gaussian blur radius (0 = no blur)
        noise: Noise level 0-1 (0 = no noise)
        align: Text alignment ('left', 'center', 'right')
        auto_size: Automatically find optimal font size to fit

    Returns:
        PIL Image (grayscale)
    """
    # Create white background
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)

    # Auto-size if requested
    if auto_size:
        font_size = find_optimal_font_size(text, font_path, size, draw)

    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try default font
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Warning: Could not load font {font_path}: {e}")
        font = ImageFont.load_default()

    # Word wrap text to fit in image
    wrapped_text = wrap_text(text, font, size - 20, draw)

    # Calculate text position based on alignment
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Vertical centering
    y = (size - text_height) // 2

    # Horizontal alignment
    if align == 'left':
        x = 10
    elif align == 'right':
        x = size - text_width - 10
    else:  # center
        x = (size - text_width) // 2

    # Draw text
    draw.multiline_text((x, y), wrapped_text, fill=0, font=font, align=align)

    # Apply rotation
    if rotation != 0:
        img = img.rotate(rotation, fillcolor=255, expand=False)

    # Apply blur
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    # Apply noise
    if noise > 0:
        img_array = np.array(img).astype(np.float32)
        noise_array = np.random.normal(0, noise * 50, img_array.shape)
        img_array = np.clip(img_array + noise_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    return img


def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def render_normal(text, size=200, font_size=20, auto_size=False, align='center'):
    """Render text in a normal, clean style (for Y/output)."""
    fonts = get_default_fonts()
    # Use first clean sans-serif font
    font_path = fonts[0] if fonts else None

    return render_text(
        text,
        size=size,
        font_path=font_path,
        font_size=font_size,
        rotation=0,
        blur=0,
        noise=0,
        align=align,
        auto_size=auto_size
    )


def render_varied(text, size=200, font_size=20, variation_level='medium', auto_size=False):
    """
    Render text with random variations (for X/input).

    Args:
        variation_level: 'low', 'medium', or 'high'
        auto_size: If True, automatically find optimal font size (overrides font_size)
    """
    fonts = get_default_fonts()

    # Random font (includes bold variants)
    font_path = np.random.choice(fonts) if fonts else None

    # Random alignment
    align = np.random.choice(['left', 'center', 'right'])

    # Variation parameters based on level
    if variation_level == 'low':
        rotation = np.random.uniform(-5, 5)
        blur = np.random.uniform(0, 0.5)
        noise = np.random.uniform(0, 0.02)
        if not auto_size:
            size_variation = np.random.uniform(0.85, 1.15)  # ±15% size
    elif variation_level == 'high':
        rotation = np.random.uniform(-20, 20)
        blur = np.random.uniform(0, 2.0)
        noise = np.random.uniform(0, 0.1)
        if not auto_size:
            size_variation = np.random.uniform(0.5, 1.5)  # ±50% size
    else:  # medium
        rotation = np.random.uniform(-10, 10)
        blur = np.random.uniform(0, 1.0)
        noise = np.random.uniform(0, 0.05)
        if not auto_size:
            size_variation = np.random.uniform(0.7, 1.3)  # ±30% size

    # Apply size variation if not auto-sizing
    if not auto_size:
        varied_font_size = int(font_size * size_variation)
    else:
        varied_font_size = font_size  # Will be ignored by auto_size

    return render_text(
        text,
        size=size,
        font_path=font_path,
        font_size=varied_font_size,
        rotation=rotation,
        blur=blur,
        noise=noise,
        align=align,
        auto_size=auto_size
    )


# Test rendering
if __name__ == "__main__":
    print("Testing text rendering...")

    test_text = "The quick brown fox jumps over the lazy dog"

    # Render normal
    img_normal = render_normal(test_text)
    img_normal.save("test_normal.png")
    print("Saved: test_normal.png")

    # Render with variations
    for i in range(3):
        img_varied = render_varied(test_text, variation_level='medium')
        img_varied.save(f"test_varied_{i}.png")
        print(f"Saved: test_varied_{i}.png")

    print("\nAvailable fonts:")
    fonts = get_default_fonts()
    for font in fonts[:5]:
        print(f"  {os.path.basename(font)}")
