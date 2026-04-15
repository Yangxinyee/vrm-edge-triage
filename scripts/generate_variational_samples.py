"""
Variational Sample Generation using MedGemma-27B.

Generates K diverse clinical interpretations per chest X-ray image
for use in MRM training. Uses temperature sampling for diversity.

Usage:
    python generate_variational_samples.py \
        --data_root data/mimic_cxr \
        --split train \
        --k_samples 5 \
        --output_file train_variational_samples.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


# System prompt for clinical report generation
SYSTEM_PROMPT = """You are a board-certified radiologist specializing in emergency diagnostics. 
Analyze the provided chest X-ray and generate a structured clinical report."""

# User instruction for report generation
USER_PROMPT = """Analyze this chest X-ray and generate a clinical report with:
1. FINDINGS: Document key positive and pertinent negative findings relevant to acute pathology
2. IMPRESSION: Provide a definitive diagnostic summary for triage classification

Use standard radiological terminology. Be concise and focus on clinically significant observations."""


class VariationalSampleGenerator:
    """
    Generate diverse clinical interpretations using MedGemma-27B.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_id: str = "google/medgemma-27b-it",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading {model_id}...")
        
        # Load model with bfloat16 for efficiency
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print("Model loaded successfully")
    
    def generate_samples(
        self,
        image: Image.Image,
        k: int = 5,
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[str]:
        """
        Generate K variational clinical interpretations for an image.
        
        Args:
            image: PIL Image of chest X-ray
            k: Number of samples to generate
            max_new_tokens: Maximum tokens per sample
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            
        Returns:
            List of K generated clinical reports
        """
        samples = []
        
        # Construct message with image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate K samples with temperature sampling
        for _ in range(k):
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.1,
                )
                output_tokens = generation[0][input_len:]
            
            decoded = self.processor.decode(output_tokens, skip_special_tokens=True)
            samples.append(decoded.strip())
        
        return samples


def load_image_ids(data_root: Path, split: str) -> List[str]:
    """Load image IDs from split list file."""
    split_file = data_root / f"{split}_list.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found")
    
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def find_image_path(images_dir: Path, image_id: str) -> Optional[Path]:
    """Find image file with given ID."""
    for ext in ['.png', '.jpg', '.jpeg']:
        path = images_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate variational samples using MedGemma-27B')
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file')
    parser.add_argument('--model_id', type=str, default='google/medgemma-27b-it')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of samples per image')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = data_root / f"{args.split}_variational_samples.json"
    
    # Load image IDs
    image_ids = load_image_ids(data_root, args.split)
    print(f"Found {len(image_ids)} images for {args.split} split")
    
    # Load existing results if resuming
    results = {}
    if args.resume and output_file.exists():
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} processed images")
    
    # Initialize generator
    generator = VariationalSampleGenerator(model_id=args.model_id)
    
    # Process images
    print(f"\nGenerating {args.k_samples} samples per image...")
    failed = []
    
    for image_id in tqdm(image_ids, desc="Processing"):
        # Skip if already processed
        if image_id in results:
            continue
        
        # Find image file
        image_path = find_image_path(images_dir, image_id)
        if image_path is None:
            failed.append(image_id)
            continue
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Generate K samples
            samples = generator.generate_samples(
                image=image,
                k=args.k_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            results[image_id] = samples
            
            # Save checkpoint every 50 images
            if len(results) % 50 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            failed.append(image_id)
            continue
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Processed: {len(results)}/{len(image_ids)} images")
    print(f"Failed: {len(failed)}")
    print(f"Output: {output_file}")
    print(f"Total samples: {len(results) * args.k_samples}")


if __name__ == '__main__':
    main()
