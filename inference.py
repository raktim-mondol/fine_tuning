"""
Inference script for fine-tuned MedGemma model on histopathology images
Based on official Google Health MedGemma implementation
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image
from transformers import pipeline, AutoProcessor
import numpy as np

from data_utils import TISSUE_CLASSES


class MedGemmaInference:
    """
    Inference class for fine-tuned MedGemma histopathology classification
    Based on official implementation using pipeline API
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to fine-tuned model directory
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.device = device
        self.tissue_classes = TISSUE_CLASSES
        
        # Load model using pipeline API (matching official implementation)
        print(f"📥 Loading model from {model_path}...")
        self.pipe = pipeline(
            "image-text-to-text",
            model=str(self.model_path),
            torch_dtype=torch.bfloat16,
        )
        
        # Set deterministic responses
        self.pipe.model.generation_config.do_sample = False
        
        # Load processor for tokenizer access
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.pipe.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # Use left padding during inference (as in official notebook)
        self.processor.tokenizer.padding_side = "left"
        
        print("✅ Model loaded successfully")
        
        # Load training metadata if available
        metadata_path = self.model_path / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"📋 Loaded training metadata")
        else:
            self.metadata = {}
    
    def predict_single_image(self, 
                           image_path: Union[str, Path, Image.Image],
                           max_new_tokens: int = 20,
                           return_confidence: bool = False) -> Dict[str, any]:
        """
        Predict tissue type for a single histopathology image
        
        Args:
            image_path: Path to image file or PIL Image object
            max_new_tokens: Maximum tokens to generate
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
            image_path_str = str(image_path)
        else:
            image = image_path.convert("RGB")
            image_path_str = "PIL_Image"
        
        # Create message format (matching official implementation)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"What is the most likely tissue type shown in the histopathology image?\n{chr(10).join(self.tissue_classes)}"}
                ]
            }
        ]
        
        # Generate prediction
        outputs = self.pipe(
            text=messages,
            images=image,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        
        prediction_text = outputs[0]["generated_text"]
        
        # Post-process prediction
        predicted_class_idx = self._postprocess_prediction(prediction_text)
        predicted_class = self.tissue_classes[predicted_class_idx] if predicted_class_idx != -1 else "Unknown"
        
        result = {
            "image_path": image_path_str,
            "prediction_text": prediction_text,
            "predicted_class": predicted_class,
            "predicted_class_idx": predicted_class_idx,
            "tissue_classes": self.tissue_classes
        }
        
        if return_confidence:
            # For now, we don't have confidence scores from the pipeline
            # This could be extended to use logits if needed
            result["confidence"] = 1.0 if predicted_class_idx != -1 else 0.0
        
        return result
    
    def predict_batch(self, 
                     image_paths: List[Union[str, Path, Image.Image]],
                     batch_size: int = 8,
                     max_new_tokens: int = 20) -> List[Dict[str, any]]:
        """
        Predict tissue types for a batch of images
        
        Args:
            image_paths: List of image paths or PIL Image objects
            batch_size: Batch size for inference
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of prediction results
        """
        print(f"🔄 Processing {len(image_paths)} images in batches of {batch_size}...")
        
        # Prepare data
        images = []
        messages_list = []
        path_strings = []
        
        for img_path in image_paths:
            # Load image
            if isinstance(img_path, (str, Path)):
                image = Image.open(img_path).convert("RGB")
                path_strings.append(str(img_path))
            else:
                image = img_path.convert("RGB")
                path_strings.append("PIL_Image")
            
            images.append(image)
            
            # Create message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"What is the most likely tissue type shown in the histopathology image?\n{chr(10).join(self.tissue_classes)}"}
                    ]
                }
            ]
            messages_list.append(messages)
        
        # Run batch inference
        outputs = self.pipe(
            text=messages_list,
            images=images,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            return_full_text=False,
        )
        
        # Process results
        results = []
        for i, output in enumerate(outputs):
            prediction_text = output["generated_text"]
            predicted_class_idx = self._postprocess_prediction(prediction_text)
            predicted_class = self.tissue_classes[predicted_class_idx] if predicted_class_idx != -1 else "Unknown"
            
            result = {
                "image_path": path_strings[i],
                "prediction_text": prediction_text,
                "predicted_class": predicted_class,
                "predicted_class_idx": predicted_class_idx,
                "tissue_classes": self.tissue_classes
            }
            results.append(result)
        
        print(f"✅ Batch processing completed")
        return results
    
    def predict_directory(self, 
                         directory_path: Union[str, Path],
                         image_extensions: List[str] = None,
                         batch_size: int = 8,
                         save_results: bool = True) -> List[Dict[str, any]]:
        """
        Predict tissue types for all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            image_extensions: List of image file extensions to process
            batch_size: Batch size for inference
            save_results: Whether to save results to JSON file
            
        Returns:
            List of prediction results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
        directory_path = Path(directory_path)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️ No image files found in {directory_path}")
            return []
        
        print(f"📁 Found {len(image_files)} images in {directory_path}")
        
        # Process images
        results = self.predict_batch(image_files, batch_size=batch_size)
        
        # Save results if requested
        if save_results:
            output_file = directory_path / "medgemma_predictions.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {output_file}")
        
        return results
    
    def _postprocess_prediction(self, prediction: str) -> int:
        """
        Convert prediction text to class index (matching evaluation implementation)
        
        Args:
            prediction: Raw prediction text from model
            
        Returns:
            Class index (-1 if no match found)
        """
        prediction_clean = prediction.strip()
        
        # Try exact match first
        for i, tissue_class in enumerate(self.tissue_classes):
            if prediction_clean == tissue_class:
                return i
        
        # Try fuzzy matching
        prediction_lower = prediction_clean.lower()
        
        # Alternative label format mapping
        alt_labels = {
            label: f"({label.replace(': ', ') ')})" for label in self.tissue_classes
        }
        
        for i, tissue_class in enumerate(self.tissue_classes):
            # Search for tissue class or alternative format in prediction
            if tissue_class in prediction_clean or alt_labels[tissue_class] in prediction_clean:
                return i
            
            # Also check if the tissue type (after colon) is mentioned
            if ': ' in tissue_class:
                tissue_type = tissue_class.split(': ')[1].lower()
                if tissue_type in prediction_lower:
                    return i
        
        return -1
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": str(self.model_path),
            "tissue_classes": self.tissue_classes,
            "num_classes": len(self.tissue_classes),
            "device": self.device,
            "metadata": self.metadata
        }
        
        return info
    
    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print("MEDGEMMA MODEL INFORMATION")
        print("="*60)
        print(f"Model Path: {info['model_path']}")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"Device: {info['device']}")
        
        if info['metadata']:
            print(f"\nTraining Metadata:")
            if 'model_id' in info['metadata']:
                print(f"  Base Model: {info['metadata']['model_id']}")
            if 'training_config' in info['metadata']:
                training_config = info['metadata']['training_config']
                print(f"  Training Epochs: {training_config.get('num_train_epochs', 'Unknown')}")
                print(f"  Learning Rate: {training_config.get('learning_rate', 'Unknown')}")
        
        print(f"\nTissue Classes:")
        for i, tissue_class in enumerate(info['tissue_classes']):
            print(f"  {i}: {tissue_class}")
        print("="*60 + "\n")


def main():
    """Main function for command-line inference"""
    parser = argparse.ArgumentParser(description="MedGemma Histopathology Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model directory")
    parser.add_argument("--image_path", type=str,
                       help="Path to single image file")
    parser.add_argument("--directory_path", type=str,
                       help="Path to directory containing images")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum tokens to generate")
    parser.add_argument("--output_file", type=str,
                       help="Path to save results (JSON format)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run inference on")
    parser.add_argument("--show_info", action="store_true",
                       help="Show model information")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = MedGemmaInference(args.model_path, device=args.device)
    
    # Show model info if requested
    if args.show_info:
        inference.print_model_info()
    
    results = []
    
    # Single image inference
    if args.image_path:
        print(f"🔍 Processing single image: {args.image_path}")
        result = inference.predict_single_image(
            args.image_path, 
            max_new_tokens=args.max_new_tokens,
            return_confidence=True
        )
        results = [result]
        
        print(f"\n📊 Prediction Results:")
        print(f"   Image: {result['image_path']}")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Raw Prediction: {result['prediction_text']}")
        if 'confidence' in result:
            print(f"   Confidence: {result['confidence']:.3f}")
    
    # Directory inference
    elif args.directory_path:
        print(f"📁 Processing directory: {args.directory_path}")
        results = inference.predict_directory(
            args.directory_path,
            batch_size=args.batch_size,
            save_results=not args.output_file  # Don't auto-save if custom output specified
        )
        
        # Print summary
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Total Images: {len(results)}")
        print(f"   Class Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"     {class_name}: {count}")
    
    else:
        print("❌ Please provide either --image_path or --directory_path")
        return
    
    # Save results to custom output file
    if args.output_file and results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
