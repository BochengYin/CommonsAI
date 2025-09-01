#!/usr/bin/env python3
"""
Script to create test images with text content for OCR validation.
Generates images with various text styles, languages, and complexities.
"""
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import json

# Test text samples in different categories
SAMPLE_TEXTS = {
    "simple": [
        "Hello World",
        "Machine Learning",
        "Data Science 101",
        "Python Programming",
        "Neural Networks"
    ],
    "technical": [
        "def neural_network(x, weights):\n    return sigmoid(x @ weights)",
        "SELECT * FROM users WHERE age > 21;",
        "git commit -m 'Add OCR functionality'",
        "import tensorflow as tf\nfrom transformers import GPT2Model",
        "HTTP 200 OK\nContent-Type: application/json"
    ],
    "complex": [
        "The Quick Brown Fox Jumps Over The Lazy Dog\n1234567890 !@#$%^&*()",
        "Multi-line text example\nwith different font sizes\nand special characters: àáâãäå",
        "Mixed CASE text with Numbers123\nSymbols: <>?{}|\\+=_)(*&^%$#@!",
        "OCR Confidence Test:\nClear Text vs ṽẻŕỳ õḇṩçũŕẻ ţẻẋţ",
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed do eiusmod tempor."
    ],
    "math": [
        "E = mc²",
        "∫₀^∞ e^(-x²) dx = √π/2",
        "f(x) = ax² + bx + c",
        "∀x ∈ ℝ: x² ≥ 0",
        "lim(x→∞) 1/x = 0"
    ],
    "multilingual": [
        "Hello / Bonjour / Hola / 你好",
        "English • Français • Español • 中文",
        "Machine Learning\nApprentissage automatique\n機械学習",
        "Data / Données / Datos / データ",
        "Algorithm / Algorithme / Algoritmo / アルゴリズム"
    ]
}

# Color schemes for different image types
COLOR_SCHEMES = {
    "high_contrast": {"bg": "white", "text": "black"},
    "dark_theme": {"bg": "#2b2b2b", "text": "#ffffff"},
    "blue_theme": {"bg": "#e8f4fd", "text": "#1565c0"},
    "sepia": {"bg": "#f4f1e8", "text": "#8b4513"},
    "low_contrast": {"bg": "#f0f0f0", "text": "#666666"}
}

def get_system_fonts():
    """Get available system fonts."""
    common_font_paths = [
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Times.ttc", 
        "/System/Library/Fonts/Courier.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Windows/Fonts/arial.ttf",
        "/Windows/Fonts/times.ttf"
    ]
    
    available_fonts = []
    for font_path in common_font_paths:
        if os.path.exists(font_path):
            available_fonts.append(font_path)
    
    return available_fonts

def create_text_image(text, width=400, height=200, font_size=20, 
                     color_scheme="high_contrast", font_path=None,
                     rotation=0, noise=False):
    """Create an image with the specified text."""
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["high_contrast"])
    
    # Create image
    img = Image.new('RGB', (width, height), color=colors["bg"])
    draw = ImageDraw.Draw(img)
    
    # Load font
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Calculate text positioning
    lines = text.split('\n')
    line_heights = []
    line_widths = []
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    
    total_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px line spacing
    max_width = max(line_widths)
    
    # Center text
    start_x = max(10, (width - max_width) // 2)
    start_y = max(10, (height - total_height) // 2)
    
    # Draw each line
    current_y = start_y
    for i, line in enumerate(lines):
        line_x = start_x
        if len(lines) > 1:  # Center each line individually for multi-line
            line_x = max(10, (width - line_widths[i]) // 2)
        
        draw.text((line_x, current_y), line, fill=colors["text"], font=font)
        current_y += line_heights[i] + 5
    
    # Apply rotation if specified
    if rotation != 0:
        img = img.rotate(rotation, fillcolor=colors["bg"], expand=False)
    
    # Add noise if specified
    if noise:
        # Add slight gaussian noise for testing OCR robustness
        import numpy as np
        img_array = np.array(img)
        noise_array = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy_array = np.clip(img_array + noise_array, 0, 255)
        img = Image.fromarray(noisy_array)
    
    return img

def generate_test_dataset(output_dir, num_per_category=5):
    """Generate a complete test dataset with various text types."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clear existing images
    for img_file in output_path.glob("*.png"):
        img_file.unlink()
        
    available_fonts = get_system_fonts()
    if not available_fonts:
        print("⚠️ No system fonts found, using default font")
        available_fonts = [None]
    
    generated_images = []
    image_counter = 1
    
    # Generate images for each category
    for category, texts in SAMPLE_TEXTS.items():
        print(f"📝 Generating {category} images...")
        
        for i in range(num_per_category):
            text = random.choice(texts)
            
            # Vary parameters
            color_scheme = random.choice(list(COLOR_SCHEMES.keys()))
            font_path = random.choice(available_fonts)
            font_size = random.choice([16, 20, 24, 28])
            width = random.choice([300, 400, 500])
            height = random.choice([150, 200, 250])
            rotation = random.choice([0, 0, 0, 2, -2, 5, -5])  # Mostly no rotation
            noise = random.choice([False, False, False, True])  # Mostly no noise
            
            # Create image
            img = create_text_image(
                text=text,
                width=width, 
                height=height,
                font_size=font_size,
                color_scheme=color_scheme,
                font_path=font_path,
                rotation=rotation,
                noise=noise
            )
            
            # Save image
            filename = f"test_{image_counter:03d}_{category}.png"
            img_path = output_path / filename
            img.save(img_path)
            
            # Record metadata
            generated_images.append({
                "filename": filename,
                "category": category,
                "text": text,
                "parameters": {
                    "width": width,
                    "height": height,
                    "font_size": font_size,
                    "color_scheme": color_scheme,
                    "font_path": font_path,
                    "rotation": rotation,
                    "noise": noise
                }
            })
            
            image_counter += 1
    
    # Save metadata
    metadata_path = output_path / "test_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_images": len(generated_images),
            "categories": list(SAMPLE_TEXTS.keys()),
            "images": generated_images
        }, f, indent=2)
    
    print(f"✅ Generated {len(generated_images)} test images in {output_dir}")
    print(f"📊 Metadata saved to {metadata_path}")
    
    return generated_images

def create_specific_test_cases(output_dir):
    """Create specific challenging test cases for OCR validation."""
    output_path = Path(output_dir)
    
    specific_cases = [
        {
            "name": "perfect_conditions",
            "text": "Perfect OCR Conditions\nClear Arial 24pt Black on White",
            "params": {"font_size": 24, "color_scheme": "high_contrast", "width": 500, "height": 150}
        },
        {
            "name": "challenging_contrast",
            "text": "Low Contrast Challenge\nGray text on light background",
            "params": {"font_size": 18, "color_scheme": "low_contrast", "width": 400, "height": 120}
        },
        {
            "name": "rotated_text", 
            "text": "Rotated Text Test\nSlightly angled content",
            "params": {"font_size": 20, "rotation": 7, "width": 400, "height": 200}
        },
        {
            "name": "small_text",
            "text": "Very Small Text Challenge for OCR Testing with multiple lines of tiny content",
            "params": {"font_size": 12, "width": 350, "height": 100}
        },
        {
            "name": "mixed_content",
            "text": "Mixed Content: TEXT123\n!@# Symbols & Numbers 456\nSpecial: àáâãäå çčć",
            "params": {"font_size": 18, "width": 400, "height": 180}
        },
        {
            "name": "code_snippet",
            "text": "def process_image(img):\n    text = ocr.extract(img)\n    return text.strip()",
            "params": {"font_size": 16, "color_scheme": "dark_theme", "width": 450, "height": 120}
        }
    ]
    
    print(f"🎯 Creating specific test cases...")
    for case in specific_cases:
        img = create_text_image(text=case["text"], **case["params"])
        filename = f"specific_{case['name']}.png"
        img.save(output_path / filename)
        print(f"   Created {filename}")
    
    return specific_cases

def validate_test_images(image_dir):
    """Validate generated test images by running OCR on them."""
    try:
        from app.ocr import get_ocr_engine
        
        engine = get_ocr_engine()
        image_path = Path(image_dir)
        
        if not image_path.exists():
            print(f"❌ Directory not found: {image_dir}")
            return
        
        test_images = list(image_path.glob("*.png"))
        if not test_images:
            print(f"❌ No PNG images found in {image_dir}")
            return
        
        print(f"\n🔍 Validating {len(test_images)} test images...")
        
        results = []
        for img_path in test_images:
            try:
                result = engine.extract_text(img_path)
                results.append({
                    "filename": img_path.name,
                    "text_length": len(result.text),
                    "confidence": result.confidence,
                    "engine": result.engine,
                    "success": len(result.text.strip()) > 0
                })
                
                if result.text.strip():
                    print(f"   ✅ {img_path.name}: {result.confidence:.2f} confidence")
                else:
                    print(f"   ⚠️ {img_path.name}: No text detected")
                    
            except Exception as e:
                print(f"   ❌ {img_path.name}: Error - {e}")
                results.append({
                    "filename": img_path.name,
                    "error": str(e),
                    "success": False
                })
        
        # Summary statistics
        successful = len([r for r in results if r.get("success", False)])
        confidences = [r["confidence"] for r in results if "confidence" in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        print(f"\n📊 Validation Summary:")
        print(f"   Successful extractions: {successful}/{len(results)}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Engines used: {set(r.get('engine', 'unknown') for r in results)}")
        
        return results
        
    except ImportError:
        print("⚠️ OCR modules not available for validation")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate test images with text content for OCR validation")
    parser.add_argument("--output-dir", default="data/images", help="Output directory for images")
    parser.add_argument("--num-per-category", type=int, default=3, help="Number of images per category")
    parser.add_argument("--validate", action="store_true", help="Run OCR validation on generated images")
    parser.add_argument("--specific-only", action="store_true", help="Only create specific test cases")
    
    args = parser.parse_args()
    
    print(f"🎨 Creating test images in {args.output_dir}")
    
    if not args.specific_only:
        # Generate main test dataset
        generated_images = generate_test_dataset(args.output_dir, args.num_per_category)
    
    # Create specific test cases
    specific_cases = create_specific_test_cases(args.output_dir)
    
    if args.validate:
        # Validate generated images
        validate_test_images(args.output_dir)
    
    print(f"\n🏁 Test image generation complete!")
    print(f"   📁 Images saved to: {args.output_dir}")
    print(f"   🔧 Next steps:")
    print(f"      1. python -m app.build_index")
    print(f"      2. uvicorn app.server:app --reload")
    print(f"      3. Test hybrid search: curl -X POST localhost:8000/search_hybrid -d 'text=machine learning'")

if __name__ == "__main__":
    main()