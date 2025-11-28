#!/usr/bin/env python3
"""
Inference script for BASE Moondream 2 model
Matches the structure of fine-tuned inference for fair comparison
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config
from src.train.dataloaders import create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseMoondream2Inference:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the base Moondream 2 model and tokenizer"""
        try:
            logger.info("Loading base Moondream2 model...")
            base_model_name = "vikhyatk/moondream2"
            md_revision = "2024-07-23"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                revision=md_revision,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "<|pad|>"
                self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                revision=md_revision,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Base model loaded and ready for inference!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image using the base model"""
        try:
            with torch.no_grad():
                enc_image = self.model.encode_image(image)
                answer = self.model.answer_question(enc_image, question, self.tokenizer)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Error"
    
    def run_inference_batch(self, batch):
        """Run inference on a batch from the dataloader"""
        pil_images = []
        
        # Load PIL images from paths (Moondream needs PIL images, not tensors)
        for img_name in batch['image_names']:
            metadata = batch['metadata'][batch['image_names'].index(img_name)]
            img_path = metadata.get('original_path', img_name)
            
            try:
                pil_img = Image.open(img_path).convert('RGB')
                pil_images.append(pil_img)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                pil_images.append(None)

        fixed_question = """
            Carefully examine this image for any of these wildfire indicators:
            - Visible flames or fire
            - Active burning
            - Glowing embers
            - Bright orange/red heat

            Answer 'Yes' only if you see clear fire. Answer 'No' for smoke alone, darkness, or unclear scenes.
            """

        # Run inference on each image in the batch
        predictions = []
        for pil_img, question in zip(pil_images, batch['questions']):
            if pil_img is not None:
                prediction = self.answer_question(pil_img, fixed_question)
                predictions.append(prediction)
            else:
                predictions.append("Error")
        
        return predictions
    
    def run_inference_on_dataloader(self, test_loader, output_file: str = None) -> Dict:
        """Run inference using dataloader"""
        logger.info(f"Running inference on {len(test_loader.dataset)} samples...")
        
        results = []
        predictions = []
        ground_truths = []
        
        # Progress bar for batches
        pbar = tqdm(test_loader, desc="Running inference", total=len(test_loader))
        
        correct_count = 0
        total_count = 0
        
        for batch in pbar:
            # Run inference on batch
            batch_predictions = self.run_inference_batch(batch)
            
            # Process batch results
            for i, prediction in enumerate(batch_predictions):
                ground_truth = batch['answers'][i]
                metadata = batch['metadata'][i]
                
                # Store result
                result = {
                    'image_name': batch['image_names'][i],
                    'question': batch['questions'][i],
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'has_fire': metadata['has_fire'],
                    'correct': self._is_correct(ground_truth, prediction)
                }
                results.append(result)
                
                # For evaluation
                predictions.append(self._normalize_answer(prediction))
                ground_truths.append(self._normalize_answer(ground_truth))
                
                # Update counters
                if result['correct']:
                    correct_count += 1
                total_count += 1
            
            # Update progress bar with running accuracy
            current_acc = correct_count / total_count * 100 if total_count > 0 else 0
            pbar.set_postfix({
                'Accuracy': f'{current_acc:.1f}%',
                'Correct': f'{correct_count}/{total_count}'
            })
        
        pbar.close()
        
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
        if 'yes' in answer:
            return 'yes'
        # if 'yes' in answer or 'fire' in answer:
        #     return 'yes'
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
        report = classification_report(ground_truths, predictions, output_dict=True, zero_division=0)
        
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
        # Create output directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
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
                   xticklabels=['Yes', 'No'], 
                   yticklabels=['Yes', 'No'])
        plt.title('Confusion Matrix - Base Moondream 2')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()


def main():
    # Dataset paths
    unified_path = unified_config["src"]
    train_json = f"{unified_path}/flame_vqa_train.json"
    val_json = f"{unified_path}/flame_vqa_val.json"
    test_json = f"{unified_path}/flame_vqa_test.json"
    
    parser = argparse.ArgumentParser(description="Run inference with base Moondream 2")
    parser.add_argument("--test_json", type=str, 
                       default=test_json,
                       help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, 
                       default="src/train/results/inference_results_base.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    parser.add_argument("--plot_cm", action="store_true",
                       help="Plot confusion matrix")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = BaseMoondream2Inference(args.device)
    
    # Create dataloader
    logger.info("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        test_json=args.test_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode='vqa',
        image_size=378,  # Match fine-tuned script
        pin_memory=True
    )
    
    # Run inference
    print("\n" + "="*80)
    print("BASE MOONDREAM 2 INFERENCE")
    print("="*80)
    
    results = inference.run_inference_on_dataloader(test_loader, args.output_file)
    
    # Print results
    metrics = results['metrics']
    print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS:")
    print(f"="*60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.1f}%)")
    print(f"")
    print(f"Fire accuracy: {metrics['fire_accuracy']:.4f} ({metrics['fire_accuracy']*100:.1f}%) - {metrics['fire_samples']} samples")
    print(f"No-fire accuracy: {metrics['no_fire_accuracy']:.4f} ({metrics['no_fire_accuracy']*100:.1f}%) - {metrics['no_fire_samples']} samples")
    print(f"="*60)
    
    # Plot confusion matrix if requested
    if args.plot_cm:
        cm_path = args.output_file.replace('.json', '_confusion_matrix.png')
        inference.plot_confusion_matrix(metrics, cm_path)


if __name__ == "__main__":
    main()
