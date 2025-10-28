#!/usr/bin/env python3
"""
Test script for the trained Moondream 2 model
This script demonstrates how to use the trained model for fire detection inference.
"""

import torch
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str):
    """Load the trained Moondream 2 model"""
    try:
        # Import Moondream 2 model class
        import sys
        sys.path.append(model_path)
        
        from moondream import Moondream
        
        # Load model
        model = Moondream.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def test_fire_detection(model, tokenizer, image_path: str, question: str = "Is there fire visible in this image?"):
    """Test fire detection on a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        
        # Generate answer
        answer = model.answer_question(image, question, tokenizer)
        
        return answer
    except Exception as e:
        logger.error(f"Error in fire detection: {e}")
        return None

def main():
    """Main test function"""
    
    # Paths
    MODEL_PATH = "./moondream2_fire_detection_custom"
    TEST_IMAGE = "/home/hice1/mchidambaram7/scratch/datasets/downloads/Training/Fire/resized_frame2093.jpg"
    
    logger.info("Loading trained model...")
    
    # Load model
    model, tokenizer = load_trained_model(MODEL_PATH)
    
    if model is None:
        logger.error("Failed to load model")
        return
    
    logger.info("Model loaded successfully!")
    
    # Test questions
    test_questions = [
        "Is there fire visible in this image?",
        "Can you detect any fire or flames in this aerial image?",
        "Does this image contain active fire?",
        "Is there any wildfire present in this scene?",
        "Are there flames visible in this thermal/RGB image?"
    ]
    
    # Test on a sample image
    if Path(TEST_IMAGE).exists():
        logger.info(f"Testing on image: {TEST_IMAGE}")
        
        for question in test_questions:
            answer = test_fire_detection(model, tokenizer, TEST_IMAGE, question)
            if answer:
                logger.info(f"Q: {question}")
                logger.info(f"A: {answer}")
                logger.info("-" * 50)
    else:
        logger.warning(f"Test image not found: {TEST_IMAGE}")
        
        # Try to find any image in the dataset
        dataset_path = Path("/home/hice1/mchidambaram7/scratch/datasets")
        possible_images = list(dataset_path.glob("**/*.jpg"))[:5]
        
        if possible_images:
            test_image = possible_images[0]
            logger.info(f"Testing on found image: {test_image}")
            
            for question in test_questions:
                answer = test_fire_detection(model, tokenizer, str(test_image), question)
                if answer:
                    logger.info(f"Q: {question}")
                    logger.info(f"A: {answer}")
                    logger.info("-" * 50)
        else:
            logger.error("No test images found")

if __name__ == "__main__":
    main()
