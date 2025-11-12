#!/usr/bin/env python3
"""
Object Detection Evaluation Script for Fine-tuned Moondream 2
Evaluates the model's ability to detect and localize fire objects in images.
"""

import torch
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import shutil
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetectionEvaluator:
    """Evaluates Moondream 2's object detection capabilities"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the evaluator with a fine-tuned model"""
        self.device = device
        self.model_path = model_path
        
        logger.info(f"Loading model from {model_path}")
        
        # Check if CUDA is actually available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
            self.device = "cpu"
        
        try:
            # Try loading fine-tuned model first
            try:
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")
                
                # Ensure custom code files are accessible
                # HuggingFace looks for custom code in cache directory
                # We need to copy custom code files to the expected cache location
                cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / model_path_obj.name
                custom_code_files = ['moondream.py', 'vision.py', 'layers.py', 'text.py', 
                                   'region.py', 'rope.py', 'utils.py', 'image_crops.py',
                                   'lora.py', 'config.py', 'hf_moondream.py']
                
                # Create cache directory if it doesn't exist
                if not cache_dir.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy custom code files from model directory to cache if they don't exist
                for code_file in custom_code_files:
                    src_file = model_path_obj / code_file
                    dst_file = cache_dir / code_file
                    if src_file.exists() and not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied {code_file} to cache directory")
                
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path_obj.absolute()),
                    trust_remote_code=True,
                    local_files_only=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model - use absolute path
                dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path_obj.absolute()),
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=True
                )
                # Move to device manually
                if device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(device)
                else:
                    self.model = self.model.to("cpu")
                    device = "cpu"
                    self.device = "cpu"
                logger.info("Fine-tuned model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model: {e}")
                logger.info("Falling back to base Moondream 2 model")
                # Fallback to base model
                base_model = "vikhyatk/moondream2"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                # Move to device manually
                if device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(device)
                else:
                    self.model = self.model.to("cpu")
                    device = "cpu"
                    self.device = "cpu"
                logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_fire_objects(self, image: Image.Image) -> Dict:
        """
        Detect fire objects in an image using the model's detect() method.
        Returns bounding boxes for detected fire objects.
        
        Note: This method may not be well-trained since fine-tuning focused on answer_question().
        """
        try:
            # Use Moondream 2's detect method to find fire objects
            result = self.model.detect(image, "fire")
            
            # The result should contain objects with bounding boxes
            if "objects" in result:
                return result["objects"]
            else:
                return []
        except Exception as e:
            logger.warning(f"Error during detection: {e}")
            return []
    
    def answer_fire_question(self, image: Image.Image, question: str = "Is there fire in this image?") -> str:
        """
        Use the fine-tuned answer_question() method to classify fire presence.
        This is what was actually fine-tuned, so it should perform better.
        """
        try:
            answer = self.model.answer_question(image, question, self.tokenizer)
            return answer.lower()
        except Exception as e:
            logger.warning(f"Error during question answering: {e}")
            return ""
    
    def normalize_bbox(self, bbox: Dict, image_width: int, image_height: int) -> Dict:
        """Normalize bounding box coordinates to [0, 1] range"""
        if "x_min" in bbox and "y_min" in bbox and "x_max" in bbox and "y_max" in bbox:
            return {
                "x_min": max(0, min(1, bbox["x_min"] / image_width)),
                "y_min": max(0, min(1, bbox["y_min"] / image_height)),
                "x_max": max(0, min(1, bbox["x_max"] / image_width)),
                "y_max": max(0, min(1, bbox["y_max"] / image_height))
            }
        return bbox
    
    def calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Normalize coordinates if needed
        x1_min, y1_min = bbox1.get("x_min", 0), bbox1.get("y_min", 0)
        x1_max, y1_max = bbox1.get("x_max", 1), bbox1.get("y_max", 1)
        x2_min, y2_min = bbox2.get("x_min", 0), bbox2.get("y_min", 0)
        x2_max, y2_max = bbox2.get("x_max", 1), bbox2.get("y_max", 1)
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def evaluate_sample(
        self, 
        image_path: str, 
        has_fire: bool,
        ground_truth_bboxes: Optional[List[Dict]] = None,
        use_vqa: bool = True
    ) -> Dict:
        """
        Evaluate object detection on a single sample.
        
        Args:
            image_path: Path to the image
            has_fire: Whether the image contains fire (ground truth)
            ground_truth_bboxes: Optional list of ground truth bounding boxes
            use_vqa: If True, also use answer_question() for classification (what was fine-tuned)
        
        Returns:
            Dictionary with evaluation metrics for this sample
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            
            # Detect fire objects using detect() method
            detected_objects = self.detect_fire_objects(image)
            
            # Normalize detected bounding boxes
            normalized_detections = []
            for obj in detected_objects:
                if "x_min" in obj:
                    normalized = self.normalize_bbox(obj, image_width, image_height)
                    normalized_detections.append(normalized)
            
            # Calculate metrics
            num_detections = len(normalized_detections)
            detection_present = num_detections > 0
            
            # Also use VQA method (what was actually fine-tuned)
            vqa_answer = ""
            vqa_classification = None
            if use_vqa:
                vqa_answer = self.answer_fire_question(image, "Is there fire in this image?")
                # Check if answer indicates fire - need to handle negative contexts
                # Check for explicit negative indicators first
                negative_indicators = ["no fire", "no flames", "no burning", "not fire", 
                                      "not a fire", "no wildfire", "no smoke", "there is no", 
                                      "there are no", "does not contain", "does not have",
                                      "no visible fire", "i don't see", "i cannot see"]
                has_negative = any(neg in vqa_answer for neg in negative_indicators)
                
                # Check for positive fire indicators
                positive_indicators = ["yes", "fire", "flame", "burning", "wildfire", "smoke"]
                has_positive = any(ind in vqa_answer for ind in positive_indicators)
                
                # Determine classification: positive indicators present AND no strong negative indicators
                vqa_classification = has_positive and not has_negative
            
            # Classification accuracy (fire detected vs not detected)
            # Use VQA if available (more accurate since it was fine-tuned)
            if vqa_classification is not None:
                classification_correct = (vqa_classification == has_fire)
                detection_method = "vqa"
            else:
                classification_correct = (detection_present == has_fire)
                detection_method = "detect"
            
            # Localization metrics (if ground truth bboxes available)
            iou_scores = []
            if ground_truth_bboxes and normalized_detections:
                # Match detections to ground truth (simple greedy matching)
                matched_gt = set()
                for det in normalized_detections:
                    best_iou = 0.0
                    best_gt_idx = None
                    for gt_idx, gt_bbox in enumerate(ground_truth_bboxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = self.calculate_iou(det, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou > 0.5:  # IoU threshold for a match
                        iou_scores.append(best_iou)
                        matched_gt.add(best_gt_idx)
            
            return {
                "image_path": image_path,
                "has_fire": has_fire,
                "num_detections": num_detections,
                "detection_present": detection_present,
                "classification_correct": classification_correct,
                "detection_method": detection_method,
                "vqa_answer": vqa_answer if use_vqa else None,
                "vqa_classification": vqa_classification if use_vqa else None,
                "detected_bboxes": normalized_detections,
                "ground_truth_bboxes": ground_truth_bboxes or [],
                "iou_scores": iou_scores,
                "mean_iou": np.mean(iou_scores) if iou_scores else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating sample {image_path}: {e}")
            return {
                "image_path": image_path,
                "has_fire": has_fire,
                "error": str(e)
            }


def load_test_samples(json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test samples from VQA JSON file"""
    logger.info(f"Loading test samples from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # Extract image path from metadata
        metadata = item.get("metadata", {})
        image_path = metadata.get("original_path")
        if not image_path or not Path(image_path).exists():
            continue
        
        # Get has_fire directly from metadata (most reliable)
        has_fire = metadata.get("has_fire", False)
        
        # Also get answer from conversations for reference
        conversations = item.get("conversations", [])
        answer = conversations[0].get("answer", "") if conversations else ""
        question = conversations[0].get("question", "") if conversations else ""
        
        samples.append({
            "image_path": image_path,
            "has_fire": has_fire,
            "question": question,
            "answer": answer,
            "metadata": metadata
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    # Log distribution of fire vs no-fire samples
    fire_count = sum(1 for s in samples if s["has_fire"])
    no_fire_count = len(samples) - fire_count
    logger.info(f"Loaded {len(samples)} valid test samples")
    logger.info(f"  Fire samples: {fire_count}, No-fire samples: {no_fire_count}")
    
    return samples


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate overall evaluation metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    # Classification metrics
    classification_correct = sum(1 for r in results if r.get("classification_correct", False))
    classification_accuracy = classification_correct / total if total > 0 else 0.0
    
    # Detection metrics
    fire_samples = [r for r in results if r.get("has_fire", False)]
    no_fire_samples = [r for r in results if not r.get("has_fire", False)]
    
    fire_detected_correctly = sum(1 for r in fire_samples if r.get("detection_present", False))
    no_fire_correctly_identified = sum(1 for r in no_fire_samples if not r.get("detection_present", False))
    
    fire_precision = fire_detected_correctly / len(fire_samples) if fire_samples else 0.0
    fire_recall = fire_detected_correctly / len(fire_samples) if fire_samples else 0.0
    
    # Average number of detections
    avg_detections = np.mean([r.get("num_detections", 0) for r in results])
    
    # IoU metrics
    all_ious = []
    for r in results:
        ious = r.get("iou_scores", [])
        all_ious.extend(ious)
    
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Confusion matrix
    true_positives = fire_detected_correctly
    false_positives = sum(1 for r in no_fire_samples if r.get("detection_present", False))
    false_negatives = len(fire_samples) - fire_detected_correctly
    true_negatives = no_fire_correctly_identified
    
    return {
        "total_samples": total,
        "classification_accuracy": classification_accuracy,
        "fire_samples": len(fire_samples),
        "no_fire_samples": len(no_fire_samples),
        "fire_detected_correctly": fire_detected_correctly,
        "no_fire_correctly_identified": no_fire_correctly_identified,
        "fire_precision": fire_precision,
        "fire_recall": fire_recall,
        "fire_f1": 2 * (fire_precision * fire_recall) / (fire_precision + fire_recall) if (fire_precision + fire_recall) > 0 else 0.0,
        "avg_detections_per_image": avg_detections,
        "mean_iou": mean_iou,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate object detection on fine-tuned Moondream 2")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/hice1/mchidambaram7/scratch/VLM-FireWatch/models/moondream2_fire_detection_best",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json",
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="object_detection_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ObjectDetectionEvaluator(args.model_path, args.device)
    
    # Load test samples
    test_samples = load_test_samples(args.test_json, args.max_samples)
    
    if not test_samples:
        logger.error("No valid test samples found")
        return
    
    # Evaluate all samples
    logger.info(f"Evaluating {len(test_samples)} samples...")
    results = []
    
    # Log a few sample answers for debugging
    logger.info("Sample ground truth answers (first 5):")
    for i, sample in enumerate(test_samples[:5]):
        logger.info(f"  Sample {i+1}: has_fire={sample['has_fire']}, answer='{sample['answer'][:100]}'")
    
    for sample in tqdm(test_samples, desc="Evaluating"):
        result = evaluator.evaluate_sample(
            sample["image_path"],
            sample["has_fire"],
            use_vqa=True  # Use VQA method which was actually fine-tuned
        )
        results.append(result)
    
    # Calculate overall metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print("\n" + "="*80)
    print("OBJECT DETECTION EVALUATION RESULTS")
    print("="*80)
    print(f"\nNOTE: Model was fine-tuned for VQA (answer_question), not object detection (detect)")
    print(f"      Using hybrid evaluation: VQA for classification + detect() for localization")
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print(f"Classification Accuracy (VQA-based): {metrics['classification_accuracy']:.4f}")
    print(f"\nFire Detection Metrics:")
    print(f"  Fire Samples: {metrics['fire_samples']}")
    print(f"  No Fire Samples: {metrics['no_fire_samples']}")
    print(f"  Fire Detected Correctly: {metrics['fire_detected_correctly']}")
    print(f"  No Fire Correctly Identified: {metrics['no_fire_correctly_identified']}")
    print(f"  Fire Precision: {metrics['fire_precision']:.4f}")
    print(f"  Fire Recall: {metrics['fire_recall']:.4f}")
    print(f"  Fire F1 Score: {metrics['fire_f1']:.4f}")
    print(f"\nLocalization Metrics (from detect() method):")
    print(f"  Average Detections per Image: {metrics['avg_detections_per_image']:.2f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Note: detect() was NOT fine-tuned, so localization may be less accurate")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Positives: {cm['true_positives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print(f"  True Negatives: {cm['true_negatives']}")
    print("="*80)
    
    # Save results
    output_data = {
        "model_path": args.model_path,
        "test_json": args.test_json,
        "metrics": metrics,
        "sample_results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()

