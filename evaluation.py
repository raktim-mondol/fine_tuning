"""
Evaluation utilities for fine-tuned MedGemma model on histopathology data
Based on official Google Health MedGemma implementation
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from PIL import Image
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
import evaluate
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import HistopathDataProcessor, TISSUE_CLASSES


class MedGemmaEvaluator:
    """
    Comprehensive evaluation suite for fine-tuned MedGemma models
    Based on official implementation
    """
    
    def __init__(self, model_dir: str, device: str = "auto"):
        """
        Initialize evaluator with fine-tuned model
        
        Args:
            model_dir: Directory containing the fine-tuned model
            device: Device to run evaluation on
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.tissue_classes = TISSUE_CLASSES
        
        # Load model using pipeline API (like official notebook)
        self.pipe = pipeline(
            "image-text-to-text",
            model=str(self.model_dir),
            torch_dtype=torch.bfloat16,
        )
        
        # Set deterministic responses
        self.pipe.model.generation_config.do_sample = False
        
        # Load processor separately for tokenizer access
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        self.pipe.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # Load training metadata if available
        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
        else:
            self.training_metadata = {}
        
        # Load evaluation metrics
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        
        print(f"✅ Model loaded from {model_dir}")
    
    def predict_single_image(self, 
                           image_path: str, 
                           max_new_tokens: int = 20) -> str:
        """
        Predict tissue type for a single image using pipeline
        
        Args:
            image_path: Path to the histopathology image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Predicted tissue type as string
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
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
        
        # Generate prediction
        outputs = self.pipe(
            text=messages,
            images=image,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        
        return outputs[0]["generated_text"]
    
    def postprocess_prediction(self, prediction: str, do_full_match: bool = False) -> int:
        """
        Convert prediction text to class index (matching official implementation)
        
        Args:
            prediction: Raw prediction text from model
            do_full_match: Whether to require exact match
            
        Returns:
            Class index (-1 if no match found)
        """
        if do_full_match:
            # Try exact match with tissue classes
            for i, tissue_class in enumerate(self.tissue_classes):
                if prediction.strip() == tissue_class:
                    return i
            return -1
        
        # Fuzzy matching
        prediction_lower = prediction.lower().strip()
        
        # Alternative label format mapping
        alt_labels = {
            label: f"({label.replace(': ', ') ')})" for label in self.tissue_classes
        }
        
        for i, tissue_class in enumerate(self.tissue_classes):
            # Search for exact tissue class or alternative format
            if tissue_class in prediction or alt_labels[tissue_class] in prediction:
                return i
        
        return -1
    
    def evaluate_dataset(self, 
                        dataset, 
                        batch_size: int = 64,
                        max_new_tokens: int = 20,
                        do_full_match: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on a dataset using batch inference
        
        Args:
            dataset: HuggingFace dataset to evaluate
            batch_size: Batch size for evaluation
            max_new_tokens: Maximum tokens to generate
            do_full_match: Whether to use exact matching
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"📊 Evaluating on {len(dataset)} samples...")
        
        # Prepare data for batch inference
        messages_list = []
        images_list = []
        ground_truth = []
        
        for sample in dataset:
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
            images_list.append(sample["image"])
            ground_truth.append(sample["label"])
        
        # Run batch inference
        print("🔄 Running batch inference...")
        outputs = self.pipe(
            text=messages_list,
            images=images_list,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            return_full_text=False,
        )
        
        # Process predictions
        predictions = []
        prediction_details = []
        
        for i, output in enumerate(outputs):
            pred_text = output["generated_text"]
            pred_idx = self.postprocess_prediction(pred_text, do_full_match)
            
            predictions.append(pred_idx)
            prediction_details.append({
                "ground_truth": ground_truth[i],
                "prediction_text": pred_text,
                "prediction_idx": pred_idx,
                "correct": pred_idx == ground_truth[i]
            })
        
        # Calculate metrics
        metrics = self.compute_metrics(predictions, ground_truth)
        
        # Add detailed predictions
        metrics["prediction_details"] = prediction_details
        metrics["num_samples"] = len(predictions)
        
        print(f"✅ Evaluation completed on {len(predictions)} samples")
        
        return metrics
    
    def compute_metrics(self, predictions: List[int], references: List[int]) -> Dict[str, float]:
        """
        Compute evaluation metrics (matching official implementation)
        
        Args:
            predictions: List of predicted class indices
            references: List of ground truth class indices
            
        Returns:
            Dictionary with accuracy and F1 metrics
        """
        # Filter out invalid predictions (-1)
        valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_references = [references[i] for i in valid_indices]
        
        if not valid_predictions:
            return {"accuracy": 0.0, "f1": 0.0, "valid_predictions": 0}
        
        metrics = {}
        
        # Compute accuracy
        metrics.update(self.accuracy_metric.compute(
            predictions=valid_predictions,
            references=valid_references,
        ))
        
        # Compute F1 score
        metrics.update(self.f1_metric.compute(
            predictions=valid_predictions,
            references=valid_references,
            average="weighted",
        ))
        
        metrics["valid_predictions"] = len(valid_predictions)
        metrics["total_predictions"] = len(predictions)
        
        return metrics
    
    def evaluate_patient_level(self, 
                              dataset,
                              aggregation_method: str = "majority_vote") -> Dict[str, Any]:
        """
        Evaluate model at patient level by aggregating patch predictions
        
        Args:
            dataset: Dataset with patient_id field
            aggregation_method: How to aggregate patch predictions
            
        Returns:
            Patient-level evaluation metrics
        """
        print(f"👥 Evaluating at patient level using {aggregation_method}...")
        
        # First get patch-level predictions
        patch_results = self.evaluate_dataset(dataset)
        
        # Group by patient
        patient_predictions = defaultdict(list)
        patient_ground_truth = {}
        
        for i, detail in enumerate(patch_results["prediction_details"]):
            if "patient_id" in dataset[i]:
                patient_id = dataset[i]["patient_id"]
                patient_predictions[patient_id].append(detail["prediction_idx"])
                patient_ground_truth[patient_id] = detail["ground_truth"]
        
        # Aggregate predictions per patient
        aggregated_predictions = []
        aggregated_references = []
        
        for patient_id, patch_preds in patient_predictions.items():
            if aggregation_method == "majority_vote":
                # Most common prediction
                valid_preds = [p for p in patch_preds if p != -1]
                if valid_preds:
                    patient_pred = max(set(valid_preds), key=valid_preds.count)
                else:
                    patient_pred = -1
            elif aggregation_method == "average_confidence":
                # This would require confidence scores - simplified to majority vote
                valid_preds = [p for p in patch_preds if p != -1]
                if valid_preds:
                    patient_pred = max(set(valid_preds), key=valid_preds.count)
                else:
                    patient_pred = -1
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            aggregated_predictions.append(patient_pred)
            aggregated_references.append(patient_ground_truth[patient_id])
        
        # Compute patient-level metrics
        patient_metrics = self.compute_metrics(aggregated_predictions, aggregated_references)
        patient_metrics["num_patients"] = len(aggregated_predictions)
        patient_metrics["aggregation_method"] = aggregation_method
        
        print(f"✅ Patient-level evaluation completed on {len(aggregated_predictions)} patients")
        
        return patient_metrics
    
    def plot_confusion_matrix(self, 
                             metrics: Dict[str, Any], 
                             save_path: str = None,
                             figsize: Tuple[int, int] = (12, 10)):
        """
        Plot confusion matrix for evaluation results
        
        Args:
            metrics: Evaluation metrics containing prediction details
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Extract predictions and ground truth
        predictions = []
        ground_truth = []
        
        for detail in metrics["prediction_details"]:
            if detail["prediction_idx"] != -1:  # Only valid predictions
                predictions.append(detail["prediction_idx"])
                ground_truth.append(detail["ground_truth"])
        
        if not predictions:
            print("⚠️ No valid predictions to plot")
            return
        
        # Create confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Create labels (shortened for display)
        labels = [tc.split(': ')[1] if ': ' in tc else tc for tc in self.tissue_classes]
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Histopathology Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def compare_with_baseline(self, 
                             test_dataset,
                             baseline_model_id: str = "google/medgemma-4b-it") -> Dict[str, Any]:
        """
        Compare fine-tuned model with baseline pretrained model
        
        Args:
            test_dataset: Test dataset for evaluation
            baseline_model_id: Baseline model identifier
            
        Returns:
            Comparison results
        """
        print(f"🔄 Comparing with baseline model: {baseline_model_id}")
        
        # Evaluate fine-tuned model
        ft_metrics = self.evaluate_dataset(test_dataset)
        
        # Load baseline model
        baseline_pipe = pipeline(
            "image-text-to-text",
            model=baseline_model_id,
            torch_dtype=torch.bfloat16,
        )
        baseline_pipe.model.generation_config.do_sample = False
        baseline_processor = AutoProcessor.from_pretrained(baseline_model_id)
        baseline_pipe.model.generation_config.pad_token_id = baseline_processor.tokenizer.eos_token_id
        
        # Evaluate baseline (similar to fine-tuned but with baseline pipeline)
        print("📊 Evaluating baseline model...")
        
        messages_list = []
        images_list = []
        ground_truth = []
        
        for sample in test_dataset:
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
            images_list.append(sample["image"])
            ground_truth.append(sample["label"])
        
        # Run baseline inference
        baseline_outputs = baseline_pipe(
            text=messages_list,
            images=images_list,
            max_new_tokens=40,  # More tokens for baseline as it might be more verbose
            batch_size=64,
            return_full_text=False,
        )
        
        # Process baseline predictions
        baseline_predictions = []
        for output in baseline_outputs:
            pred_text = output["generated_text"]
            pred_idx = self.postprocess_prediction(pred_text, do_full_match=False)  # Use fuzzy matching for baseline
            baseline_predictions.append(pred_idx)
        
        # Compute baseline metrics
        baseline_metrics = self.compute_metrics(baseline_predictions, ground_truth)
        
        # Create comparison
        comparison = {
            "fine_tuned": {
                "accuracy": ft_metrics["accuracy"],
                "f1": ft_metrics["f1"],
                "valid_predictions": ft_metrics["valid_predictions"]
            },
            "baseline": {
                "accuracy": baseline_metrics["accuracy"],
                "f1": baseline_metrics["f1"],
                "valid_predictions": baseline_metrics["valid_predictions"]
            },
            "improvement": {
                "accuracy": ft_metrics["accuracy"] - baseline_metrics["accuracy"],
                "f1": ft_metrics["f1"] - baseline_metrics["f1"]
            }
        }
        
        print(f"📈 Comparison Results:")
        print(f"   Fine-tuned - Accuracy: {ft_metrics['accuracy']:.3f}, F1: {ft_metrics['f1']:.3f}")
        print(f"   Baseline - Accuracy: {baseline_metrics['accuracy']:.3f}, F1: {baseline_metrics['f1']:.3f}")
        print(f"   Improvement - Accuracy: {comparison['improvement']['accuracy']:.3f}, F1: {comparison['improvement']['f1']:.3f}")
        
        return comparison
    
    def generate_evaluation_report(self, 
                                  test_dataset,
                                  output_dir: str = "./evaluation_results",
                                  include_baseline_comparison: bool = True):
        """
        Generate comprehensive evaluation report
        
        Args:
            test_dataset: Test dataset for evaluation
            output_dir: Directory to save results
            include_baseline_comparison: Whether to compare with baseline
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📋 Generating comprehensive evaluation report...")
        
        # Patch-level evaluation
        patch_metrics = self.evaluate_dataset(test_dataset)
        
        # Patient-level evaluation (if patient_id available)
        patient_metrics = None
        if len(test_dataset) > 0 and "patient_id" in test_dataset[0]:
            patient_metrics = self.evaluate_patient_level(test_dataset)
        
        # Baseline comparison
        comparison = None
        if include_baseline_comparison:
            try:
                comparison = self.compare_with_baseline(test_dataset)
            except Exception as e:
                print(f"⚠️ Baseline comparison failed: {e}")
        
        # Generate plots
        self.plot_confusion_matrix(
            patch_metrics, 
            save_path=output_dir / "confusion_matrix.png"
        )
        
        # Save detailed results
        results = {
            "model_info": {
                "model_dir": str(self.model_dir),
                "tissue_classes": self.tissue_classes,
                "training_metadata": self.training_metadata
            },
            "patch_level_metrics": patch_metrics,
            "patient_level_metrics": patient_metrics,
            "baseline_comparison": comparison,
            "evaluation_config": {
                "test_samples": len(test_dataset),
                "timestamp": str(np.datetime64('now'))
            }
        }
        
        # Save results to JSON
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate text summary
        summary = self._generate_text_summary(patch_metrics, patient_metrics, comparison)
        with open(output_dir / "evaluation_summary.txt", 'w') as f:
            f.write(summary)
        
        print(f"✅ Evaluation report saved to {output_dir}")
        print(f"📊 Key Results:")
        print(f"   Patch-level Accuracy: {patch_metrics['accuracy']:.3f}")
        print(f"   Patch-level F1: {patch_metrics['f1']:.3f}")
        if patient_metrics:
            print(f"   Patient-level Accuracy: {patient_metrics['accuracy']:.3f}")
            print(f"   Patient-level F1: {patient_metrics['f1']:.3f}")
    
    def _generate_text_summary(self, 
                              patch_metrics: Dict[str, Any], 
                              patient_metrics: Dict[str, Any] = None,
                              comparison: Dict[str, Any] = None) -> str:
        """Generate text summary of evaluation results"""
        
        summary = "MedGemma Histopathology Classification - Evaluation Summary\n"
        summary += "=" * 60 + "\n\n"
        
        # Model info
        summary += f"Model Directory: {self.model_dir}\n"
        summary += f"Tissue Classes: {len(self.tissue_classes)}\n\n"
        
        # Patch-level results
        summary += "Patch-Level Results:\n"
        summary += f"  Accuracy: {patch_metrics['accuracy']:.4f}\n"
        summary += f"  F1 Score: {patch_metrics['f1']:.4f}\n"
        summary += f"  Valid Predictions: {patch_metrics['valid_predictions']}/{patch_metrics['total_predictions']}\n\n"
        
        # Patient-level results
        if patient_metrics:
            summary += "Patient-Level Results:\n"
            summary += f"  Accuracy: {patient_metrics['accuracy']:.4f}\n"
            summary += f"  F1 Score: {patient_metrics['f1']:.4f}\n"
            summary += f"  Patients Evaluated: {patient_metrics['num_patients']}\n"
            summary += f"  Aggregation Method: {patient_metrics['aggregation_method']}\n\n"
        
        # Baseline comparison
        if comparison:
            summary += "Baseline Comparison:\n"
            summary += f"  Fine-tuned Accuracy: {comparison['fine_tuned']['accuracy']:.4f}\n"
            summary += f"  Baseline Accuracy: {comparison['baseline']['accuracy']:.4f}\n"
            summary += f"  Accuracy Improvement: {comparison['improvement']['accuracy']:.4f}\n"
            summary += f"  Fine-tuned F1: {comparison['fine_tuned']['f1']:.4f}\n"
            summary += f"  Baseline F1: {comparison['baseline']['f1']:.4f}\n"
            summary += f"  F1 Improvement: {comparison['improvement']['f1']:.4f}\n\n"
        
        # Tissue class breakdown
        summary += "Tissue Classes:\n"
        for i, tissue_class in enumerate(self.tissue_classes):
            summary += f"  {i}: {tissue_class}\n"
        
        return summary


if __name__ == "__main__":
    # Example usage
    evaluator = MedGemmaEvaluator("./medgemma-4b-it-sft-lora-histopath")
    
    # Load test dataset (you would load your actual test data here)
    print("📝 Example usage - replace with your actual test dataset")
    print("evaluator.generate_evaluation_report(test_dataset)")
