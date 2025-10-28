#!/usr/bin/env python3
"""
Simple inference script for Moondream 2 on fire detection images.
No training required - just test the base model's capabilities.
"""

import torch
import json
import random
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_moondream2_model(device):
    """Load Moondream 2 model and tokenizer"""
    model_name = "vikhyatk/moondream2"
    
    try:
        logger.info(f"Loading Moondream 2 model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device.type == 'cuda' else None
        )
        
        logger.info("Moondream 2 model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def load_test_samples(json_path):
    """Load all valid samples from the dataset for testing"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Filter samples that have existing images
        valid_samples = []
        for item in data:
            image_path = item['metadata']['original_path']
            if Path(image_path).exists():
                valid_samples.append(item)
        
        logger.info(f"Found {len(valid_samples)} valid samples out of {len(data)} total")
        return valid_samples
        
    except Exception as e:
        logger.error(f"Error loading test samples: {e}")
        return []

def run_inference(model, tokenizer, image_path, question):
    """Run inference on a single image and question"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Get model's answer
        answer = model.answer_question(image, question, tokenizer)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return f"Error: {e}"

def main():
    """Main inference function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_moondream2_model(device)
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Load test samples
    test_json = "/home/hice1/mchidambaram7/scratch/datasets/unified_dataset/flame_vqa_test.json"
    logger.info(f"Loading test samples from: {test_json}")
    
    test_samples = load_test_samples(test_json)
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    if not test_samples:
        logger.error("No valid test samples found")
        return
    
    # Test questions
    test_questions = [
        "Is there fire visible in this image?",
        "Can you detect any fire or flames in this aerial image?",
        "Does this image contain active fire?",
        "Are there any flames or burning areas in this image?",
        "What do you see in this image?",
        "Is this a fire scene?"
    ]
    
    # Run inference on all test samples
    logger.info("Running inference on all test samples...")
    print("\n" + "="*80)
    print("MOONDREAM 2 FIRE DETECTION INFERENCE RESULTS")
    print("="*80)
    
    correct_predictions = 0
    total_predictions = 0
    fire_correct = 0
    fire_total = 0
    no_fire_correct = 0
    no_fire_total = 0
    
    # Use the first question for evaluation
    evaluation_question = test_questions[0]
    
    print(f"Evaluation Question: {evaluation_question}")
    print(f"Processing {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        if i % 100 == 0:  # Progress update every 100 samples
            print(f"Processing sample {i+1}/{len(test_samples)}...")
        
        # Get image info
        image_path = sample['metadata']['original_path']
        has_fire = sample['metadata']['has_fire']
        source_dataset = sample['metadata']['source_dataset']
        
        # Run inference
        answer = run_inference(model, tokenizer, image_path, evaluation_question)
        
        # Simple evaluation (keyword matching)
        answer_lower = answer.lower()
        fire_keywords = ['fire', 'flame', 'burning', 'yes']
        no_fire_keywords = ['no fire', 'no flames', 'no burning', 'no']
        
        model_has_fire = any(keyword in answer_lower for keyword in fire_keywords)
        model_no_fire = any(keyword in answer_lower for keyword in no_fire_keywords)
        
        # Count predictions
        if (has_fire and model_has_fire) or (not has_fire and model_no_fire):
            correct_predictions += 1
        total_predictions += 1
        
        # Count by category
        if has_fire:
            fire_total += 1
            if model_has_fire:
                fire_correct += 1
        else:
            no_fire_total += 1
            if model_no_fire:
                no_fire_correct += 1
    
    # Summary
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        fire_accuracy = fire_correct / fire_total * 100 if fire_total > 0 else 0
        no_fire_accuracy = no_fire_correct / no_fire_total * 100 if no_fire_total > 0 else 0
        
        print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS:")
        print(f"="*60)
        print(f"Total Samples Processed: {total_predictions}")
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        print(f"")
        print(f"Fire Detection Accuracy: {fire_accuracy:.1f}% ({fire_correct}/{fire_total})")
        print(f"No-Fire Detection Accuracy: {no_fire_accuracy:.1f}% ({no_fire_correct}/{no_fire_total})")
        print(f"")
        print(f"Dataset Breakdown:")
        print(f"  - Fire samples: {fire_total}")
        print(f"  - No-fire samples: {no_fire_total}")
        
        # Save results to file
        results = {
            "total_samples": total_predictions,
            "overall_accuracy": accuracy,
            "fire_accuracy": fire_accuracy,
            "no_fire_accuracy": no_fire_accuracy,
            "fire_correct": fire_correct,
            "fire_total": fire_total,
            "no_fire_correct": no_fire_correct,
            "no_fire_total": no_fire_total,
            "evaluation_question": evaluation_question
        }
        
        results_file = "/home/hice1/mchidambaram7/scratch/VLM-FireWatch/moondream2_inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*80)
    print("Inference completed!")

if __name__ == "__main__":
    main()
