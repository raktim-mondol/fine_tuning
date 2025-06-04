"""
Utility script for inference on new histopathology images
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from PIL import Image

from evaluation import MedGemmaEvaluator


def batch_inference(model_dir: str, 
                   image_paths: List[str],
                   output_file: str = None,
                   temperature: float = 0.1,
                   max_new_tokens: int = 50) -> List[Dict[str, Any]]:
    """
    Run inference on multiple images
    
    Args:
        model_dir: Directory containing fine-tuned model
        image_paths: List of paths to histopathology images
        output_file: Optional output file for results
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of prediction results
    """
    print(f"🔬 Running inference on {len(image_paths)} images...")
    
    # Initialize evaluator
    evaluator = MedGemmaEvaluator(model_dir)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"📸 Processing {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        try:
            # Run prediction
            prediction = evaluator.predict_single_image(
                image_path=image_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            
            result = {
                "image_path": str(image_path),
                "image_name": Path(image_path).name,
                "prediction": prediction,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            result = {
                "image_path": str(image_path),
                "image_name": Path(image_path).name,
                "prediction": None,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_file}")
    
    # Print summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n📊 Inference Summary:")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Failed: {len(results) - successful}/{len(results)}")
    
    return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="Run inference on histopathology images using fine-tuned MedGemma"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing fine-tuned model"
    )
    
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Paths to histopathology images"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing histopathology images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0 for deterministic)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
        help="Image file extensions to process"
    )
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    
    if args.images:
        # Use specified image paths
        image_paths.extend(args.images)
    
    if args.image_dir:
        # Collect images from directory
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        for ext in args.extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        raise ValueError("No images specified. Use --images or --image_dir")
    
    # Convert to strings and filter existing files
    valid_paths = []
    for path in image_paths:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            valid_paths.append(str(path))
        else:
            print(f"⚠️ Skipping non-existent file: {path}")
    
    if not valid_paths:
        raise ValueError("No valid image files found")
    
    print(f"🔍 Found {len(valid_paths)} valid images")
    
    # Run inference
    results = batch_inference(
        model_dir=args.model_dir,
        image_paths=valid_paths,
        output_file=args.output,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )
    
    # Print some example results
    print("\n📋 Example Results:")
    for result in results[:5]:  # Show first 5 results
        status_icon = "✅" if result["status"] == "success" else "❌"
        prediction = result.get("prediction", "Error")
        print(f"   {status_icon} {result['image_name']}: {prediction}")
    
    if len(results) > 5:
        print(f"   ... and {len(results) - 5} more results")


if __name__ == "__main__":
    main()
