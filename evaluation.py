"""
Evaluation utilities for fine-tuned MedGemma model on histopathology data
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import HistopathDataProcessor


class MedGemmaEvaluator:
    """
    Comprehensive evaluation suite for fine-tuned MedGemma models
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
        
        # Load model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        
        # Load training metadata if available
        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
        else:
            self.training_metadata = {}
        
        self.model.eval()
        print(f"✅ Model loaded from {model_dir}")
    
    def predict_single_image(self, 
                           image_path: str, 
                           temperature: float = 0.1,
                           max_new_tokens: int = 50) -> str:
        """
        Predict histopathology subtype for a single image
        
        Args:
            image_path: Path to the histopathology image
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Predicted subtype as string
        """
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        input_text = "Classify the histopathology subtype in this image:"
        
        # Prepare inputs
        inputs = self.processor(
            text=input_text,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode result
        prediction = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return prediction
    
    def evaluate_dataset(self, 
                        dataset, 
                        batch_size: int = 8,
                        temperature: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate model on a dataset
        
        Args:
            dataset: HuggingFace dataset to evaluate
            batch_size: Batch size for evaluation
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"📊 Evaluating on {len(dataset)} samples...")
        
        predictions = []
        ground_truth = []
        prediction_details = []
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Process each sample in batch
            for j, sample in enumerate(batch):
                if (i + j) % 50 == 0:
                    print(f"   Progress: {i + j}/{len(dataset)}")
                
                try:
                    # Get prediction
                    pred = self.predict_single_image(
                        sample["image_path"] if "image_path" in sample else sample["image"],
                        temperature=temperature
                    )
                    
                    predictions.append(pred)
                    ground_truth.append(sample["subtype"])
                    
                    prediction_details.append({
                        "image_path": sample.get("image_path", ""),
                        "patient_id": sample.get("patient_id", ""),
                        "ground_truth": sample["subtype"],
                        "prediction": pred,
                        "correct": self._is_prediction_correct(pred, sample["subtype"])
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing sample {i + j}: {e}")
                    continue
        
        # Calculate metrics
        metrics = self.calculate_metrics(ground_truth, predictions)
        
        # Add detailed predictions
        metrics["prediction_details"] = prediction_details
        metrics["num_samples"] = len(predictions)
        
        print(f"✅ Evaluation completed on {len(predictions)} samples")
        
        return metrics
    
    def _is_prediction_correct(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth
        Uses fuzzy matching to handle variations in model output
        """
        pred_lower = prediction.lower().strip()
        gt_lower = ground_truth.lower().strip()
        
        # Exact match
        if pred_lower == gt_lower:
            return True
        
        # Check if ground truth is contained in prediction
        if gt_lower in pred_lower:
            return True
        
        # Check for common variations
        # You can extend this based on your specific classes
        variations = {
            "adenocarcinoma": ["adeno", "adenoca"],
            "squamous cell carcinoma": ["squamous", "scc", "squamous cell"],
            "normal": ["normal tissue", "benign", "healthy"],
            "inflammatory": ["inflammation", "inflam"]
        }
        
        for canonical, variants in variations.items():
            if canonical.lower() == gt_lower:
                for variant in variants:
                    if variant in pred_lower:
                        return True
        
        return False
    
    def calculate_metrics(self, 
                         ground_truth: List[str], 
                         predictions: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            ground_truth: List of true labels
            predictions: List of predicted labels
            
        Returns:
            Dictionary containing various metrics
        """
        # Create label mappings
        unique_labels = sorted(list(set(ground_truth)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert to indices for sklearn metrics
        true_indices = [label_to_idx[gt] for gt in ground_truth]
        pred_indices = []
        
        for pred in predictions:
            # Find best matching label
            matched_idx = self._find_best_matching_label(pred, unique_labels, label_to_idx)
            pred_indices.append(matched_idx)
        
        # Calculate metrics
        accuracy = accuracy_score(true_indices, pred_indices)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_indices, pred_indices, average=None, labels=range(len(unique_labels))
        )
        
        # Calculate macro and weighted averages
        f1_macro = f1_score(true_indices, pred_indices, average='macro')
        f1_weighted = f1_score(true_indices, pred_indices, average='weighted')
        
        # Classification report
        class_report = classification_report(
            true_indices,
            pred_indices,
            target_names=unique_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_indices, pred_indices)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(unique_labels):
            per_class_metrics[label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i]
            }
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_metrics": per_class_metrics,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "label_names": unique_labels,
            "label_mapping": label_to_idx
        }
    
    def _find_best_matching_label(self, 
                                 prediction: str, 
                                 unique_labels: List[str], 
                                 label_to_idx: Dict[str, int]) -> int:
        """Find the best matching label for a prediction"""
        pred_lower = prediction.lower().strip()
        
        # First try exact matches
        for label in unique_labels:
            if self._is_prediction_correct(prediction, label):
                return label_to_idx[label]
        
        # If no match found, return -1 or map to most common class
        # Here we'll map to the first class as fallback
        return 0 if unique_labels else -1
    
    def evaluate_patient_level(self, 
                              dataset,
                              aggregation_method: str = "majority_vote") -> Dict[str, Any]:
        """
        Evaluate model at patient level by aggregating patch predictions
        
        Args:
            dataset: Dataset with patient_id information
            aggregation_method: How to aggregate patch predictions
                              ("majority_vote", "confidence_weighted", "all_agree")
            
        Returns:
            Patient-level evaluation metrics
        """
        print(f"👥 Evaluating at patient level using {aggregation_method}...")
        
        # Group predictions by patient
        patient_predictions = defaultdict(list)
        patient_ground_truth = {}
        
        for sample in dataset:
            patient_id = sample.get("patient_id", "unknown")
            
            # Get prediction for this patch
            pred = self.predict_single_image(
                sample["image_path"] if "image_path" in sample else sample["image"]
            )
            
            patient_predictions[patient_id].append(pred)
            patient_ground_truth[patient_id] = sample["subtype"]  # Assuming same for all patches
        
        # Aggregate predictions per patient
        final_predictions = []
        final_ground_truth = []
        
        for patient_id, patch_preds in patient_predictions.items():
            gt = patient_ground_truth[patient_id]
            
            if aggregation_method == "majority_vote":
                # Count predictions and take majority
                pred_counts = defaultdict(int)
                for pred in patch_preds:
                    # Find best matching known class
                    matched_class = self._match_to_known_class(pred, list(set(patient_ground_truth.values())))
                    pred_counts[matched_class] += 1
                
                final_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
                
            elif aggregation_method == "all_agree":
                # Only consider correct if all patches agree
                unique_preds = set(patch_preds)
                if len(unique_preds) == 1:
                    final_pred = list(unique_preds)[0]
                else:
                    final_pred = "disagreement"
            
            else:  # Default to first prediction
                final_pred = patch_preds[0] if patch_preds else "unknown"
            
            final_predictions.append(final_pred)
            final_ground_truth.append(gt)
        
        # Calculate patient-level metrics
        patient_metrics = self.calculate_metrics(final_ground_truth, final_predictions)
        patient_metrics["num_patients"] = len(final_predictions)
        patient_metrics["aggregation_method"] = aggregation_method
        
        print(f"✅ Patient-level evaluation completed on {len(final_predictions)} patients")
        
        return patient_metrics
    
    def _match_to_known_class(self, prediction: str, known_classes: List[str]) -> str:
        """Match prediction to the most likely known class"""
        for known_class in known_classes:
            if self._is_prediction_correct(prediction, known_class):
                return known_class
        
        # Fallback to first known class
        return known_classes[0] if known_classes else "unknown"
    
    def plot_confusion_matrix(self, 
                             metrics: Dict[str, Any], 
                             save_path: str = None,
                             figsize: Tuple[int, int] = (10, 8)):
        """
        Plot confusion matrix
        
        Args:
            metrics: Metrics dictionary containing confusion matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        conf_matrix = np.array(metrics["confusion_matrix"])
        labels = metrics["label_names"]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, 
                                  test_dataset,
                                  output_dir: str = "./evaluation_results"):
        """
        Generate comprehensive evaluation report
        
        Args:
            test_dataset: Test dataset to evaluate on
            output_dir: Directory to save evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("📋 Generating comprehensive evaluation report...")
        
        # Patch-level evaluation
        patch_metrics = self.evaluate_dataset(test_dataset)
        
        # Patient-level evaluation (if patient_id available)
        patient_metrics = None
        if "patient_id" in test_dataset[0]:
            patient_metrics = self.evaluate_patient_level(test_dataset)
        
        # Save results
        results = {
            "patch_level_metrics": patch_metrics,
            "patient_level_metrics": patient_metrics,
            "model_info": {
                "model_dir": str(self.model_dir),
                "training_metadata": self.training_metadata
            }
        }
        
        # Save JSON report
        with open(output_dir / "evaluation_report.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Plot confusion matrix
        if patch_metrics.get("confusion_matrix") is not None:
            self.plot_confusion_matrix(
                patch_metrics, 
                str(output_dir / "confusion_matrix_patch.png")
            )
        
        if patient_metrics and patient_metrics.get("confusion_matrix") is not None:
            self.plot_confusion_matrix(
                patient_metrics,
                str(output_dir / "confusion_matrix_patient.png")
            )
        
        # Generate text summary
        summary_text = self._generate_text_summary(patch_metrics, patient_metrics)
        with open(output_dir / "evaluation_summary.txt", 'w') as f:
            f.write(summary_text)
        
        print(f"📊 Evaluation report saved to {output_dir}")
        print(f"📈 Patch-level accuracy: {patch_metrics['accuracy']:.4f}")
        if patient_metrics:
            print(f"👥 Patient-level accuracy: {patient_metrics['accuracy']:.4f}")
        
        return results
    
    def _generate_text_summary(self, 
                              patch_metrics: Dict[str, Any], 
                              patient_metrics: Dict[str, Any] = None) -> str:
        """Generate text summary of evaluation results"""
        summary = []
        summary.append("=" * 80)
        summary.append("MEDGEMMA HISTOPATHOLOGY EVALUATION REPORT")
        summary.append("=" * 80)
        summary.append("")
        
        # Model info
        summary.append("MODEL INFORMATION:")
        summary.append(f"  Model Directory: {self.model_dir}")
        if self.training_metadata:
            summary.append(f"  Base Model: {self.training_metadata.get('model_id', 'Unknown')}")
            summary.append(f"  Training Epochs: {self.training_metadata.get('training_args', {}).get('num_epochs', 'Unknown')}")
        summary.append("")
        
        # Patch-level results
        summary.append("PATCH-LEVEL EVALUATION:")
        summary.append(f"  Number of samples: {patch_metrics['num_samples']}")
        summary.append(f"  Accuracy: {patch_metrics['accuracy']:.4f}")
        summary.append(f"  F1 (Macro): {patch_metrics['f1_macro']:.4f}")
        summary.append(f"  F1 (Weighted): {patch_metrics['f1_weighted']:.4f}")
        summary.append("")
        
        summary.append("  Per-class metrics:")
        for class_name, metrics in patch_metrics['per_class_metrics'].items():
            summary.append(f"    {class_name}:")
            summary.append(f"      Precision: {metrics['precision']:.4f}")
            summary.append(f"      Recall: {metrics['recall']:.4f}")
            summary.append(f"      F1: {metrics['f1']:.4f}")
            summary.append(f"      Support: {metrics['support']}")
        summary.append("")
        
        # Patient-level results
        if patient_metrics:
            summary.append("PATIENT-LEVEL EVALUATION:")
            summary.append(f"  Number of patients: {patient_metrics['num_patients']}")
            summary.append(f"  Aggregation method: {patient_metrics['aggregation_method']}")
            summary.append(f"  Accuracy: {patient_metrics['accuracy']:.4f}")
            summary.append(f"  F1 (Macro): {patient_metrics['f1_macro']:.4f}")
            summary.append(f"  F1 (Weighted): {patient_metrics['f1_weighted']:.4f}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset
    
    # Initialize evaluator
    evaluator = MedGemmaEvaluator("./medgemma-histpath-finetuned")
    
    # Example: load test dataset and evaluate
    # test_dataset = load_dataset("your_test_dataset")["test"]
    # results = evaluator.generate_evaluation_report(test_dataset)
