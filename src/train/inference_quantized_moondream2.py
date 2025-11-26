#!/usr/bin/env python3
"""
Inference script for quantized Moondream 2 model
Uses TensorRT INT8 engine for vision encoder + PyTorch text model
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

# TensorRT imports
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config, moondream_config
from src.train.dataloaders import create_dataloaders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTVisionEncoder:
    """TensorRT wrapper for quantized vision encoder"""
    
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.input_shape = (1, 3, 378, 378)
        self.output_shape = (1, 729, 2048)
        
        # Load engine
        self._load_engine()
        
        # Allocate buffers (fixed type casting)
        input_size = int(np.prod(self.input_shape) * 4)  # ← FIX
        output_size = int(np.prod(self.output_shape) * 4)  # ← FIX
        
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()

    def _load_engine(self):
        """Load TensorRT engine from file"""
        logger.info(f"Loading TensorRT engine from: {self.engine_path}")
        
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        logger.info("✓ TensorRT engine loaded successfully")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess PIL image to numpy array (same as Moondream2)"""
        # Resize to 378x378
        img = image.resize((378, 378))
        
        # Convert to numpy and normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Moondream normalization
        
        # Transpose HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
         # ← FIX: Make array contiguous
        return np.ascontiguousarray(img, dtype=np.float32)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Run TensorRT inference on image"""
        # Preprocess image
        input_data = self.preprocess(image)
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Run inference - FIX: Use execute_v2 instead of execute_async_v2
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_v2(bindings=bindings)  # ← CHANGED
        
        # Copy output from device
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        self.stream.synchronize()
        
        # Convert to PyTorch tensor
        return torch.from_numpy(output_data).to(torch.float16)


class QuantizedMoondream2Inference:
    def __init__(self, model_path: str, trt_engine_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.trt_engine_path = trt_engine_path
        self.model = None
        self.tokenizer = None
        self.trt_vision_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load quantized vision encoder (TensorRT) + fine-tuned text model (PyTorch)"""
        try:
            # Step 1: Load TensorRT vision encoder
            logger.info("Loading TensorRT quantized vision encoder...")
            self.trt_vision_encoder = TensorRTVisionEncoder(self.trt_engine_path)
            
            # Step 2: Load base model architecture from HuggingFace
            logger.info("Loading base Moondream2 text model...")
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
            
            # Step 3: Load fine-tuned text model weights
            logger.info("Loading fine-tuned text model weights...")
            finetuned_weights_path = os.path.join(self.model_path, "pytorch_model.bin")
            
            if os.path.exists(finetuned_weights_path):
                state_dict = torch.load(finetuned_weights_path, map_location='cpu')
                # Only load text model weights (vision encoder is replaced by TensorRT)
                text_model_state = {k: v for k, v in state_dict.items() if 'text_model' in k}
                self.model.load_state_dict(text_model_state, strict=False)
                logger.info("Fine-tuned text weights loaded successfully!")
            else:
                # Try alternative weight file names
                alt_paths = [
                    os.path.join(self.model_path, "model.safetensors"),
                    os.path.join(self.model_path, "pytorch_model.safetensors"),
                ]
                
                loaded = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        from safetensors.torch import load_file
                        state_dict = load_file(alt_path)
                        text_model_state = {k: v for k, v in state_dict.items() if 'text_model' in k}
                        self.model.load_state_dict(text_model_state, strict=False)
                        logger.info(f"Fine-tuned text weights loaded from {alt_path}!")
                        loaded = True
                        break
                
                if not loaded:
                    logger.warning("No fine-tuned weights found! Using base text model.")
            
            # Step 4: Move text model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Quantized model loaded and ready for inference!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using TensorRT vision encoder"""
        # Use TensorRT instead of PyTorch vision encoder
        image_embeddings = self.trt_vision_encoder(image)
        return image_embeddings.to(self.device)
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        try:
            with torch.no_grad():
                # Encode image with TensorRT
                enc_image = self.encode_image(image)
                # Use text model to answer (same as before)
                answer = self.model.answer_question(enc_image, question, self.tokenizer)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Error"
    
    def run_inference_batch(self, batch):
        """Run inference on a batch from the dataloader"""
        pil_images = []
        
        # Load PIL images from paths
        for img_name in batch['image_names']:
            metadata = batch['metadata'][batch['image_names'].index(img_name)]
            img_path = metadata.get('original_path', img_name)
            
            try:
                pil_img = Image.open(img_path).convert('RGB')
                pil_images.append(pil_img)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                pil_images.append(None)
        
        # Run inference on each image
        predictions = []
        for pil_img, question in zip(pil_images, batch['questions']):
            if pil_img is not None:
                prediction = self.answer_question(pil_img, question)
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
        if 'yes' in answer or 'fire' in answer:
            return 'yes'
        elif 'no' in answer or 'not' in answer:
            return 'no'
        else:
            return answer
    
    def _calculate_metrics(self, ground_truths: List[str], predictions: List[str], results: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        overall_accuracy = accuracy_score(ground_truths, predictions)
        
        fire_results = [r for r in results if r['has_fire']]
        no_fire_results = [r for r in results if not r['has_fire']]
        
        fire_accuracy = sum(r['correct'] for r in fire_results) / len(fire_results) if fire_results else 0
        no_fire_accuracy = sum(r['correct'] for r in no_fire_results) / len(no_fire_results) if no_fire_results else 0
        
        report = classification_report(ground_truths, predictions, output_dict=True, zero_division=0)
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
        plt.title('Confusion Matrix - Quantized Moondream 2')
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
    train_json = os.path.join(unified_path, "flame_vqa_train.json")
    val_json = os.path.join(unified_path, "flame_vqa_val.json")
    test_json = os.path.join(unified_path, "flame_vqa_test.json")
    
    tuned_moondream_path = moondream_config["fine_tuned"]
    trt_engine_path = moondream_config["quantized"]
    
    parser = argparse.ArgumentParser(description="Run inference with quantized Moondream 2")
    parser.add_argument("--model_path", type=str, default=tuned_moondream_path, 
                       help="Path to fine-tuned model directory")
    parser.add_argument("--trt_engine", type=str, default=trt_engine_path,
                       help="Path to TensorRT engine file")
    parser.add_argument("--test_json", type=str, 
                       default=test_json,
                       help="Path to test JSON file")
    parser.add_argument("--output_file", type=str, 
                       default="src/train/results/inference_results_quantized.json",
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
    
    # Check paths exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    if not os.path.exists(args.trt_engine):
        logger.error(f"TensorRT engine not found: {args.trt_engine}")
        return
    
    # Initialize inference
    inference = QuantizedMoondream2Inference(args.model_path, args.trt_engine, args.device)
    
    # Create dataloader
    logger.info("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        test_json=args.test_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode='vqa',
        image_size=378,
        pin_memory=True
    )
    
    # Run inference
    print("\n" + "="*80)
    print("QUANTIZED MOONDREAM 2 INFERENCE (TensorRT INT8)")
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
