#!/usr/bin/env python3
"""
Inference script for fine-tuned Moondream 2 model
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinetunedMoondream2Inference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading fine-tuned model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map={"": self.device}
            ).to(self.device)
            
            logger.info("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image using the fine-tuned model"""
        try:
            with torch.no_grad():
                answer = self.model.answer_question(image, question, self.tokenizer)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Error"
    
    def run_inference_on_dataset(self, test_json: str, output_file: str = None) -> Dict:
        """Run inference on entire test dataset"""
        logger.info(f"Loading test dataset from: {test_json}")
        
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        
        logger.info(f"Running inference on {len(test_data)} samples...")
        
        results = []
        predictions = []
        ground_truths = []
        
        for i, sample in enumerate(tqdm(test_data, desc="Running inference")):
            try:
                # Load image
                image_path = sample['metadata']['original_path']
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                image = Image.open(image_path).convert('RGB')
                image = image.resize((224, 224))
                
                # Get question and ground truth
                question = sample['conversations'][0]['question']
                ground_truth = sample['conversations'][0]['answer']
                
                # Run inference
                prediction = self.answer_question(image, question)
                
                # Store results
                result = {
                    'image_name': sample['image'],
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'has_fire': sample['metadata']['has_fire'],
                    'correct': self._is_correct(ground_truth, prediction)
                }
                results.append(result)
                
                # For evaluation
                predictions.append(self._normalize_answer(prediction))
                ground_truths.append(self._normalize_answer(ground_truth))
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(ground_truths, predictions, results)
        
        # Save results
        if output_file:
            self._save_results(results, metrics, output_file)
        
        return {
            'results': results,
            'metrics': metrics,
            'total_samples': len(results)
        }
    
    def _is_correct(self, ground_truth: str, prediction: str) -> bool:
        """Check if prediction is correct"""
        gt_norm = self._normalize_answer(ground_truth)
        pred_norm = self._normalize_answer(prediction)
        return gt_norm == pred_norm
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.lower().strip()
        if 'yes' in answer or 'fire' in answer:
            return 'yes'
        elif 'no' in answer or 'not' in answer:
            return 'no'
        else:
            return answer
    
    def _calculate_metrics(self, ground_truths: List[str], predictions: List[str], results: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        # Overall accuracy
        overall_accuracy = accuracy_score(ground_truths, predictions)
        
        # Fire vs No-fire accuracy
        fire_results = [r for r in results if r['has_fire']]
        no_fire_results = [r for r in results if not r['has_fire']]
        
        fire_accuracy = sum(r['correct'] for r in fire_results) / len(fire_results) if fire_results else 0
        no_fire_accuracy = sum(r['correct'] for r in no_fire_results) / len(no_fire_results) if no_fire_results else 0
        
        # Classification report
        report = classification_report(ground_truths, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions, labels=['yes', 'no'])
        
        return {
            'overall_accuracy': overall_accuracy,
            'fire_accuracy': fire_accuracy,
            'no_fire_accuracy': no_fire_accuracy,
            'fire_samples': len(fire_results),
            'no_fire_samples': len(no_fire_results),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def _save_results(self, results: List[Dict], metrics: Dict, output_file: str):
        """Save inference results and metrics"""
        output_data = {
            'metrics': metrics,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: str = None):
        """Plot confusion matrix"""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Fire', 'Fire'], 
                   yticklabels=['No Fire', 'Fire'])
        plt.title('Confusion Matrix - Fine-tuned Moondream 2')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Moondream 2")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to fine-tuned model directory")
    parser.add_argument("--test_json", type=str, 
                       default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json",
                       help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, 
                       default="inference_results_finetuned.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--plot_cm", action="store_true",
                       help="Plot confusion matrix")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Initialize inference
    inference = FinetunedMoondream2Inference(args.model_path, args.device)
    
    # Run inference
    results = inference.run_inference_on_dataset(args.test_json, args.output_file)
    
    # Print results
    metrics = results['metrics']
    logger.info("\n" + "="*50)
    logger.info("INFERENCE RESULTS")
    logger.info("="*50)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"Fire accuracy: {metrics['fire_accuracy']:.4f} ({metrics['fire_samples']} samples)")
    logger.info(f"No-fire accuracy: {metrics['no_fire_accuracy']:.4f} ({metrics['no_fire_samples']} samples)")
    
    # Plot confusion matrix if requested
    if args.plot_cm:
        cm_path = args.output_file.replace('.json', '_confusion_matrix.png')
        inference.plot_confusion_matrix(metrics, cm_path)

if __name__ == "__main__":
    main()
