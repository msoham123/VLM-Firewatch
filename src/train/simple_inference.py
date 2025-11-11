#!/usr/bin/env python3
"""
Simple inference script for fine-tuned Moondream 2 model
"""

import os
import json
import argparse
import logging
from typing import List, Dict
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finetuned_model(model_path: str, device: str = "cuda"):
    """Load the fine-tuned model and tokenizer"""
    try:
        logger.info(f"Loading fine-tuned model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": device},
            local_files_only=True
        ).to(device)
        
        logger.info("Fine-tuned model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        logger.info("Trying to load original model instead...")
        
        # Fallback to original model
        model_name = "vikhyatk/moondream2"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": device}
        ).to(device)
        
        logger.info("Original model loaded successfully!")
        return model, tokenizer

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.lower().strip()
    if 'yes' in answer or 'fire' in answer:
        return 'yes'
    elif 'no' in answer or 'not' in answer:
        return 'no'
    else:
        return answer

def is_correct(ground_truth: str, prediction: str) -> bool:
    """Check if prediction is correct"""
    gt_norm = normalize_answer(ground_truth)
    pred_norm = normalize_answer(prediction)
    return gt_norm == pred_norm

def run_inference(model, tokenizer, test_json: str, output_file: str = None, device: str = "cuda"):
    """Run inference on test dataset"""
    logger.info(f"Loading test dataset from: {test_json}")
    
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Running inference on {len(test_data)} samples...")
    
    results = []
    correct_count = 0
    fire_correct = 0
    no_fire_correct = 0
    fire_total = 0
    no_fire_total = 0
    
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
            has_fire = sample['metadata']['has_fire']
            
            # Run inference
            with torch.no_grad():
                prediction = model.answer_question(image, question, tokenizer)
            
            # Check correctness
            correct = is_correct(ground_truth, prediction)
            if correct:
                correct_count += 1
                if has_fire:
                    fire_correct += 1
                else:
                    no_fire_correct += 1
            
            if has_fire:
                fire_total += 1
            else:
                no_fire_total += 1
            
            # Store result
            result = {
                'image_name': sample['image'],
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'has_fire': has_fire,
                'correct': correct
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    # Calculate metrics
    total_samples = len(results)
    overall_accuracy = correct_count / total_samples if total_samples > 0 else 0
    fire_accuracy = fire_correct / fire_total if fire_total > 0 else 0
    no_fire_accuracy = no_fire_correct / no_fire_total if no_fire_total > 0 else 0
    
    metrics = {
        'total_samples': total_samples,
        'overall_accuracy': overall_accuracy,
        'fire_accuracy': fire_accuracy,
        'no_fire_accuracy': no_fire_accuracy,
        'fire_samples': fire_total,
        'no_fire_samples': no_fire_total,
        'fire_correct': fire_correct,
        'no_fire_correct': no_fire_correct
    }
    
    # Save results
    if output_file:
        output_data = {
            'metrics': metrics,
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Moondream 2")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to fine-tuned model directory")
    parser.add_argument("--test_json", type=str, 
                       default="/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json",
                       help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, 
                       default="simple_inference_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Load model
    model, tokenizer = load_finetuned_model(args.model_path, args.device)
    
    # Run inference
    metrics, results = run_inference(model, tokenizer, args.test_json, args.output_file, args.device)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("INFERENCE RESULTS")
    logger.info("="*50)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"Fire accuracy: {metrics['fire_accuracy']:.4f} ({metrics['fire_correct']}/{metrics['fire_samples']} samples)")
    logger.info(f"No-fire accuracy: {metrics['no_fire_accuracy']:.4f} ({metrics['no_fire_correct']}/{metrics['no_fire_samples']} samples)")
    
    # Show some example predictions
    logger.info("\n" + "="*50)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("="*50)
    for i, result in enumerate(results[:5]):  # Show first 5 results
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Question: {result['question']}")
        logger.info(f"  Ground Truth: {result['ground_truth']}")
        logger.info(f"  Prediction: {result['prediction']}")
        logger.info(f"  Correct: {result['correct']}")
        logger.info(f"  Has Fire: {result['has_fire']}")
        logger.info("")

if __name__ == "__main__":
    main()
