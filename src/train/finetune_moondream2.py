# #!/usr/bin/env python3
# """
# Moondream 2 Fine-tuning Script for Fire Detection
# Based on official Moondream2 fine-tuning notebook
# """

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from PIL import Image
# import json
# import os
# import logging
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# from tqdm import tqdm
# import wandb
# from datetime import datetime
# import argparse
# import math

# try:
#     from bitsandbytes.optim import Adam8bit
# except ImportError:
#     print("Warning: bitsandbytes not available, using standard AdamW")
#     Adam8bit = None

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from src.data.dataset_configs import unified_config, moondream_config

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Constants from official implementation
# ANSWER_EOS = "<|endoftext|>"
# IMG_TOKENS = 729  # Number of tokens used to represent each image


# class Moondream2FireDataset(Dataset):
#     """
#     Custom dataset for Moondream 2 fine-tuning on fire detection VQA data
#     Returns data in the format expected by official Moondream2 training
#     """
    
#     def __init__(
#         self, 
#         json_path: str,
#         max_samples: Optional[int] = None
#     ):
#         # Load data
#         logger.info(f"Loading dataset from {json_path}")
#         with open(json_path, 'r') as f:
#             self.data = json.load(f)
        
#         # Filter valid samples
#         self.valid_data = []
#         for item in tqdm(self.data, desc="Validating samples"):
#             if self._validate_sample(item):
#                 self.valid_data.append(item)
        
#         if max_samples:
#             self.valid_data = self.valid_data[:max_samples]
        
#         logger.info(f"Loaded {len(self.valid_data)} valid samples")
    
#     def _validate_sample(self, item: Dict) -> bool:
#         """Validate that a sample has all required fields and image exists"""
#         try:
#             if 'conversations' not in item or 'metadata' not in item:
#                 return False
            
#             image_path = item['metadata']['original_path']
#             if not os.path.exists(image_path):
#                 return False
            
#             if not item['conversations'] or len(item['conversations']) < 1:
#                 return False
            
#             qa_pair = item['conversations'][0]
#             if 'question' not in qa_pair or 'answer' not in qa_pair:
#                 return False
            
#             return True
#         except Exception:
#             return False
    
#     def __len__(self) -> int:
#         return len(self.valid_data)
    
#     def __getitem__(self, idx: int) -> Dict:
#         item = self.valid_data[idx]
        
#         # Load image as PIL
#         image_path = item['metadata']['original_path']
#         image = Image.open(image_path).convert('RGB')
        
#         # Get Q&A pairs (format as list like official notebook)
#         qa_list = []
#         for qa_pair in item['conversations']:
#             qa_list.append({
#                 'question': qa_pair['question'],
#                 'answer': qa_pair['answer']
#             })
        
#         return {
#             'image': image,  # PIL Image
#             'qa': qa_list,  # List of Q&A dicts
#             'has_fire': item['metadata']['has_fire']
#         }


# class Moondream2Collator:
#     """
#     Custom collator matching official Moondream2 fine-tuning approach
#     """
    
#     def __init__(self, moondream_model, tokenizer):
#         self.moondream = moondream_model
#         self.tokenizer = tokenizer
    
#     def __call__(self, batch: List[Dict]) -> Tuple:
#         # Preprocess images using official method
#         images = [sample['image'] for sample in batch]
#         images = [self.moondream.vision_encoder.preprocess(image) for image in images]
        
#         labels_acc = []
#         tokens_acc = []
        
#         # Process each sample following official approach
#         for sample in batch:
#             # Start with BOS token
#             toks = [self.tokenizer.bos_token_id]
#             # Reserve space for image tokens + BOS (set labels to -100 for no loss)
#             labs = [-100] * (IMG_TOKENS + 1)
            
#             # Process each Q&A pair
#             for qa in sample['qa']:
#                 # Tokenize question (no loss on question tokens)
#                 q_t = self.tokenizer(
#                     f"\n\nQuestion: {qa['question']}\n\nAnswer:",
#                     add_special_tokens=False
#                 ).input_ids
#                 toks.extend(q_t)
#                 labs.extend([-100] * len(q_t))
                
#                 # Tokenize answer (compute loss on answer tokens)
#                 a_t = self.tokenizer(
#                     f" {qa['answer']}{ANSWER_EOS}",
#                     add_special_tokens=False
#                 ).input_ids
#                 toks.extend(a_t)
#                 labs.extend(a_t)  # Loss computed on answer tokens
            
#             tokens_acc.append(toks)
#             labels_acc.append(labs)
        
#         # Pad to max length in batch
#         max_len = max(len(labels) for labels in labels_acc)
        
#         attn_mask_acc = []
#         for i in range(len(batch)):
#             len_i = len(labels_acc[i])
#             pad_i = max_len - len_i
            
#             # Pad with -100 for labels, eos_token for tokens
#             labels_acc[i].extend([-100] * pad_i)
#             tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
#             attn_mask_acc.append([1] * len_i + [0] * pad_i)
        
#         return (
#             images,
#             torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
#             torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
#             torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
#         )


# class Moondream2FineTuner:
#     """
#     Main fine-tuning class for Moondream 2 (following official approach)
#     """
    
#     def __init__(
#         self,
#         model_name: str = "vikhyatk/moondream2",
#         device: str = "cuda",
#         dtype: torch.dtype = torch.float16,
#         md_revision: str = "2024-07-23"
#     ):
#         self.model_name = model_name
#         self.device = device
#         self.dtype = torch.float32 if device == "cpu" else dtype
#         self.md_revision = md_revision
        
#         # Load model and tokenizer
#         self._load_model_and_tokenizer()
        
#         # Initialize optimizer and scheduler
#         self.optimizer = None
#         self.scheduler = None
    
#     def _load_model_and_tokenizer(self):
#         """Load Moondream 2 model and tokenizer"""
#         logger.info(f"Loading model: {self.model_name}")
        
#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             revision=self.md_revision
#         )
        
#         # Load model with official settings
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             revision=self.md_revision,
#             trust_remote_code=True,
#             attn_implementation="flash_attention_2" if self.device == "cuda" else None,
#             torch_dtype=self.dtype,
#         )

#         # Move model to device (ADD THIS)
#         self.model = self.model.to(self.device)
        
#         logger.info("Model and tokenizer loaded successfully")
    
#     def setup_training(
#         self,
#         learning_rate: float = 1e-5,
#         total_steps: int = 1000,
#         grad_accum_steps: int = 2
#     ):
#         """Setup training: freeze vision encoder, enable gradient checkpointing"""
        
#         # Only train text model (vision encoder stays frozen)
#         self.model.text_model.train()
#         self.model.vision_encoder.eval()
        
#         # Enable gradient checkpointing for memory efficiency
#         if hasattr(self.model.text_model, 'transformer'):
#             self.model.text_model.transformer.gradient_checkpointing_enable()
        
#         # Setup optimizer (only optimize text model parameters)
#         if Adam8bit is not None:
#             self.optimizer = Adam8bit(
#                 [{"params": self.model.text_model.parameters()}],
#                 lr=learning_rate * 0.1,  # Start at 0.1 * LR
#                 betas=(0.9, 0.95),
#                 eps=1e-6
#             )
#         else:
#             self.optimizer = torch.optim.AdamW(
#                 self.model.text_model.parameters(),
#                 lr=learning_rate * 0.1,
#                 betas=(0.9, 0.95),
#                 eps=1e-6
#             )
        
#         self.total_steps = total_steps
#         self.base_lr = learning_rate
#         self.grad_accum_steps = grad_accum_steps
        
#         logger.info(f"Training setup complete. LR={learning_rate}, Steps={total_steps}")
    
#     def lr_schedule(self, step: int) -> float:
#         """
#         Custom LR schedule from official implementation:
#         - Warmup from 0.1*LR to LR over first 10% of training
#         - Cosine decay from LR to 0.1*LR over remaining 90%
#         """
#         x = step / self.total_steps
#         if x < 0.1:
#             return 0.1 * self.base_lr + 0.9 * self.base_lr * x / 0.1
#         else:
#             return 0.1 * self.base_lr + 0.9 * self.base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2
    
#     def compute_loss(self, batch: Tuple) -> torch.Tensor:
#         """
#         Compute loss following official approach:
#         - Encode images with frozen vision encoder (no_grad)
#         - Get token embeddings
#         - Concatenate: [BOS embedding, image embeddings, text embeddings]
#         - Forward through text model
#         """
#         images, tokens, labels, attn_mask = batch
        
#         # Move tokens to device
#         tokens = tokens.to(self.device)
#         labels = labels.to(self.device)
#         attn_mask = attn_mask.to(self.device)
        
#         # Encode images with frozen vision encoder (process individually)
#         with torch.no_grad():
#             # Move each image to device and encode
#             img_embs_list = []
#             for img in images:
#                 # Ensure tensor is on GPU and contiguous
#                 if not isinstance(img, torch.Tensor):
#                     img = torch.tensor(img)
#                 img_cuda = img.to(self.device, dtype=self.dtype).contiguous()
#                 img_emb = self.model.vision_encoder(img_cuda.unsqueeze(0))  # Add batch dim
#                 img_embs_list.append(img_emb)
#             img_embs = torch.cat(img_embs_list, dim=0)  # Concatenate along batch dimension
        
#         # Get token embeddings
#         tok_embs = self.model.text_model.get_input_embeddings()(tokens)
        
#         # Concatenate: [first token (BOS), image embeddings, remaining tokens]
#         inputs_embeds = torch.cat((
#             tok_embs[:, 0:1, :],  # BOS token
#             img_embs,              # Image embeddings (729 tokens)
#             tok_embs[:, 1:, :]    # Text tokens
#         ), dim=1)
        
#         # Forward through text model
#         outputs = self.model.text_model(
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             attention_mask=attn_mask,
#             use_cache=False
#         )
        
#         return outputs.loss

#     def train_epoch(
#         self,
#         train_loader: DataLoader,
#         epoch: int,
#         use_wandb: bool = True
#     ) -> float:
#         """Train for one epoch"""
        
#         total_loss = 0
#         step = 0
        
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
#         for batch_idx, batch in enumerate(progress_bar):
#             step += 1
            
#             # Compute loss
#             loss = self.compute_loss(batch)
#             loss.backward()
            
#             # Update weights after gradient accumulation
#             if step % self.grad_accum_steps == 0:
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.model.text_model.parameters(), max_norm=1.0)
                
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
                
#                 # Update learning rate
#                 lr = self.lr_schedule(step // self.grad_accum_steps)
#                 for param_group in self.optimizer.param_groups:
#                     param_group['lr'] = lr
                
#                 # Log
#                 if use_wandb:
#                     wandb.log({
#                         "loss/train": loss.item(),
#                         "lr": lr,
#                         "epoch": epoch,
#                         "step": step // self.grad_accum_steps
#                     })
            
#             total_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
        
#         return total_loss / len(train_loader)
    
#     def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
#         """Validate the model"""
        
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
#                 loss = self.compute_loss(batch)
#                 total_loss += loss.item()
        
#         avg_loss = total_loss / len(val_loader)
        
#         self.model.text_model.train()  # Back to training mode
        
#         logger.info(f"Validation Loss: {avg_loss:.4f}")
        
#         return avg_loss, 0.0  # Return dummy accuracy for now
    
#     def save_model(self, save_path: str):
#         """Save the fine-tuned model"""
#         logger.info(f"Saving model to {save_path}")
        
#         os.makedirs(save_path, exist_ok=True)
        
#         # Save model and tokenizer
#         self.model.save_pretrained(save_path)
#         self.tokenizer.save_pretrained(save_path)
        
#         logger.info("Model saved successfully")


# def create_dataloaders(
#     train_json: str,
#     val_json: str,
#     moondream_model,
#     tokenizer,
#     batch_size: int = 8,
#     num_workers: int = 2,  # Set to 0 to avoid multiprocessing issues
#     max_train_samples: Optional[int] = None,
#     max_val_samples: Optional[int] = None
# ) -> Tuple[DataLoader, DataLoader]:
#     """Create training and validation dataloaders"""
    
#     # Create datasets
#     train_dataset = Moondream2FireDataset(
#         json_path=train_json,
#         max_samples=max_train_samples
#     )
    
#     val_dataset = Moondream2FireDataset(
#         json_path=val_json,
#         max_samples=max_val_samples
#     )
    
#     # Create collator
#     collator = Moondream2Collator(moondream_model, tokenizer)
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=collator,
#         pin_memory=True if num_workers > 0 else False
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         collate_fn=collator,
#         pin_memory=True if num_workers > 0 else False
#     )
    
#     logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
#     return train_loader, val_loader


# def main():
#     """Main fine-tuning function"""
    
#     # Parse arguments
#     parser = argparse.ArgumentParser(description="Fine-tune Moondream 2 on fire detection")
#     parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
#     parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
#     parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
#     parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
#     parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
#     parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
#     parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
#     parser.add_argument("--test_mode", action="store_true", help="Test mode with small dataset")
    
#     args = parser.parse_args()
    
#     # Device setup
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float16 if device == "cuda" else torch.float32
    
#     logger.info(f"Using device: {device}, dtype: {dtype}")
    
#     # Test mode adjustments
#     if args.test_mode:
#         args.max_train_samples = 50
#         args.max_val_samples = 10
#         args.epochs = 1
#         args.batch_size = 2
#         logger.info("Running in test mode with limited samples")
    
#     # Initialize wandb
#     if args.use_wandb:
#         wandb.init(
#             project="moondream2-fire-detection",
#             config={
#                 "batch_size": args.batch_size,
#                 "grad_accum_steps": args.grad_accum_steps,
#                 "epochs": args.epochs,
#                 "learning_rate": args.learning_rate,
#                 "device": device,
#                 "dtype": str(dtype)
#             }
#         )
    
#     # Dataset paths
#     unified_path = unified_config["src"]
#     train_json = f"{unified_path}/flame_vqa_train.json"
#     val_json = f"{unified_path}/flame_vqa_val.json"
    
#     # Initialize fine-tuner
#     fine_tuner = Moondream2FineTuner(device=device, dtype=dtype)
    
#     # Create dataloaders
#     train_loader, val_loader = create_dataloaders(
#         train_json=train_json,
#         val_json=val_json,
#         moondream_model=fine_tuner.model,
#         tokenizer=fine_tuner.tokenizer,
#         batch_size=args.batch_size,
#         max_train_samples=args.max_train_samples,
#         max_val_samples=args.max_val_samples
#     )
    
#     # Setup training
#     total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
#     fine_tuner.setup_training(
#         learning_rate=args.learning_rate,
#         total_steps=total_steps,
#         grad_accum_steps=args.grad_accum_steps
#     )
    
#     # Training loop
#     logger.info("Starting fine-tuning...")
#     best_val_loss = float('inf')
    
#     for epoch in range(1, args.epochs + 1):
#         # Train
#         train_loss = fine_tuner.train_epoch(
#             train_loader,
#             epoch,
#             use_wandb=args.use_wandb
#         )
        
#         # Validate
#         val_loss, val_acc = fine_tuner.validate(val_loader, epoch)
        
#         logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
#         # Log to wandb
#         if args.use_wandb:
#             wandb.log({
#                 "epoch": epoch,
#                 "train_loss_epoch": train_loss,
#                 "val_loss": val_loss
#             })
        
#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             save_path = moondream_config["fine_tuned"]
#             fine_tuner.save_model(save_path)
#             logger.info(f"New best model saved with val_loss={val_loss:.4f}")
    
#     # Save final model
#     final_save_path = moondream_config["fine_tuned"]
#     fine_tuner.save_model(final_save_path)
    
#     logger.info("Fine-tuning completed!")
    
#     if args.use_wandb:
#         wandb.finish()

# if __name__ == "__main__":
#     main()


"""
Moondream 2 Fine-tuning Script for Fire Detection with LoRA
Based on official Moondream2 fine-tuning notebook
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb
from datetime import datetime
import argparse
import math

try:
    from bitsandbytes.optim import Adam8bit
except ImportError:
    print("Warning: bitsandbytes not available, using standard AdamW")
    Adam8bit = None

# ← ADD: LoRA/PEFT imports
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.dataset_configs import unified_config, moondream_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from official implementation
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729

class Moondream2FireDataset(Dataset):
    """
    Custom dataset for Moondream 2 fine-tuning on fire detection VQA data
    Returns data in the format expected by official Moondream2 training
    """
    
    def __init__(
        self, 
        json_path: str,
        max_samples: Optional[int] = None
    ):
        logger.info(f"Loading dataset from {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.valid_data = []
        for item in tqdm(self.data, desc="Validating samples"):
            if self._validate_sample(item):
                self.valid_data.append(item)
        
        if max_samples:
            self.valid_data = self.valid_data[:max_samples]
        
        logger.info(f"Loaded {len(self.valid_data)} valid samples")
    
    def _validate_sample(self, item: Dict) -> bool:
        try:
            if 'conversations' not in item or 'metadata' not in item:
                return False
            
            image_path = item['metadata']['original_path']
            if not os.path.exists(image_path):
                return False
            
            if not item['conversations'] or len(item['conversations']) < 1:
                return False
            
            qa_pair = item['conversations'][0]
            if 'question' not in qa_pair or 'answer' not in qa_pair:
                return False
            
            return True
        except Exception:
            return False
    
    def __len__(self) -> int:
        return len(self.valid_data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.valid_data[idx]
        
        image_path = item['metadata']['original_path']
        image = Image.open(image_path).convert('RGB')
        
        qa_list = []
        for qa_pair in item['conversations']:
            qa_list.append({
                'question': qa_pair['question'],
                'answer': qa_pair['answer']
            })
        
        return {
            'image': image,
            'qa': qa_list,
            'has_fire': item['metadata']['has_fire']
        }

class Moondream2Collator:
    """
    Custom collator matching official Moondream2 fine-tuning approach
    """
    
    def __init__(self, moondream_model, tokenizer):
        self.moondream = moondream_model
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict]) -> Tuple:
        images = [sample['image'] for sample in batch]
        images = [self.moondream.vision_encoder.preprocess(image) for image in images]
        
        labels_acc = []
        tokens_acc = []
        
        for sample in batch:
            toks = [self.tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)
            
            for qa in sample['qa']:
                q_t = self.tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))
                
                a_t = self.tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)
            
            tokens_acc.append(toks)
            labels_acc.append(labs)
        
        max_len = max(len(labels) for labels in labels_acc)
        
        attn_mask_acc = []
        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i
            
            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([self.tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)
        
        return (
            images,
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

class Moondream2FineTuner:
    """
    Main fine-tuning class for Moondream 2 with LoRA support
    """
    
    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        md_revision: str = "2024-07-23",
        use_lora: bool = True,  # ← ADD: LoRA flag
        lora_r: int = 8,  # ← ADD: LoRA rank
        lora_alpha: int = 16  # ← ADD: LoRA alpha
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = torch.float32 if device == "cpu" else dtype
        self.md_revision = md_revision
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        self._load_model_and_tokenizer()
        
        # ← ADD: Apply LoRA if enabled
        if self.use_lora and PEFT_AVAILABLE:
            self._apply_lora()
        
        self.optimizer = None
        self.scheduler = None
    
    def _load_model_and_tokenizer(self):
        """Load Moondream 2 model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.md_revision
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.md_revision,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None,
            torch_dtype=self.dtype,
        )
        
        self.model = self.model.to(self.device)
        logger.info("Model and tokenizer loaded successfully")
    
    def _apply_lora(self):
        """Apply LoRA to text model for parameter-efficient fine-tuning"""
        logger.info("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["Wqkv", "out_proj", "fc1", "fc2"],  # ← FIXED: Correct Phi layer names
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model.text_model = get_peft_model(self.model.text_model, lora_config)
        self.model.text_model.print_trainable_parameters()
        
        logger.info("✓ LoRA applied successfully!")
    
    def setup_training(
        self,
        learning_rate: float = 5e-6,  # ← CHANGED: Lower default LR
        total_steps: int = 1000,
        grad_accum_steps: int = 2
    ):
        """Setup training: freeze vision encoder, enable gradient checkpointing"""
        
        self.model.text_model.train()
        self.model.vision_encoder.eval()
        
        # ← FIX: Only enable gradient checkpointing if NOT using LoRA
        if hasattr(self.model.text_model, 'transformer') and not self.use_lora:
            self.model.text_model.transformer.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif self.use_lora:
            logger.info("Gradient checkpointing disabled (incompatible with LoRA)")
        
        # ← MODIFIED: Get trainable parameters (works for both LoRA and full fine-tuning)
        trainable_params = [p for p in self.model.text_model.parameters() if p.requires_grad]
        
        if Adam8bit is not None:
            self.optimizer = Adam8bit(
                trainable_params,
                lr=learning_rate * 0.1,
                betas=(0.9, 0.95),
                eps=1e-6
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate * 0.1,
                betas=(0.9, 0.95),
                eps=1e-6
            )
        
        self.total_steps = total_steps
        self.base_lr = learning_rate
        self.grad_accum_steps = grad_accum_steps
        
        logger.info(f"Training setup complete. LR={learning_rate}, Steps={total_steps}")
    
    def lr_schedule(self, step: int) -> float:
        """LR schedule: warmup + cosine decay"""
        x = step / self.total_steps
        if x < 0.1:
            return 0.1 * self.base_lr + 0.9 * self.base_lr * x / 0.1
        else:
            return 0.1 * self.base_lr + 0.9 * self.base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2
    
    def compute_loss(self, batch: Tuple) -> torch.Tensor:
        """Compute loss following official approach"""
        images, tokens, labels, attn_mask = batch
        
        tokens = tokens.to(self.device)
        labels = labels.to(self.device)
        attn_mask = attn_mask.to(self.device)
        
        with torch.no_grad():
            img_embs_list = []
            for img in images:
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                img_cuda = img.to(self.device, dtype=self.dtype).contiguous()
                img_emb = self.model.vision_encoder(img_cuda.unsqueeze(0))
                img_embs_list.append(img_emb)
            img_embs = torch.cat(img_embs_list, dim=0)
        
        tok_embs = self.model.text_model.get_input_embeddings()(tokens)
        
        inputs_embeds = torch.cat((
            tok_embs[:, 0:1, :],
            img_embs,
            tok_embs[:, 1:, :]
        ), dim=1)
        
        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
            use_cache=False
        )
        
        return outputs.loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        use_wandb: bool = True
    ) -> float:
        """Train for one epoch"""
        
        total_loss = 0
        step = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step += 1
            
            loss = self.compute_loss(batch)
            loss.backward()
            
            if step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.text_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                lr = self.lr_schedule(step // self.grad_accum_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                if use_wandb:
                    wandb.log({
                        "loss/train": loss.item(),
                        "lr": lr,
                        "epoch": epoch,
                        "step": step // self.grad_accum_steps
                    })
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        
        self.model.text_model.train()
        
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss, 0.0
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model"""
        logger.info(f"Saving model to {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("Model saved successfully")

def create_dataloaders(
    train_json: str,
    val_json: str,
    moondream_model,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 2,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    train_dataset = Moondream2FireDataset(
        json_path=train_json,
        max_samples=max_train_samples
    )
    
    val_dataset = Moondream2FireDataset(
        json_path=val_json,
        max_samples=max_val_samples
    )
    
    collator = Moondream2Collator(moondream_model, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if num_workers > 0 else False
    )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    """Main fine-tuning function"""
    
    parser = argparse.ArgumentParser(description="Fine-tune Moondream 2 on fire detection")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")  # ← CHANGED
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with small dataset")
    # ← ADD: LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Using device: {device}, dtype: {dtype}")
    logger.info(f"LoRA enabled: {args.use_lora}")  # ← ADD
    
    if args.test_mode:
        args.max_train_samples = 50
        args.max_val_samples = 10
        args.epochs = 1
        args.batch_size = 2
        logger.info("Running in test mode with limited samples")
    
    if args.use_wandb:
        wandb.init(
            project="moondream2-fire-detection-lora",  # ← CHANGED: New project name
            config={
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "use_lora": args.use_lora,  # ← ADD
                "lora_r": args.lora_r,  # ← ADD
                "device": device,
                "dtype": str(dtype)
            }
        )
    
    unified_path = unified_config["src"]
    train_json = f"{unified_path}/flame_vqa_train.json"
    val_json = f"{unified_path}/flame_vqa_val.json"
    
    # ← MODIFIED: Pass LoRA parameters
    fine_tuner = Moondream2FineTuner(
        device=device,
        dtype=dtype,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    train_loader, val_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        moondream_model=fine_tuner.model,
        tokenizer=fine_tuner.tokenizer,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    fine_tuner.setup_training(
        learning_rate=args.learning_rate,
        total_steps=total_steps,
        grad_accum_steps=args.grad_accum_steps
    )
    
    logger.info("Starting fine-tuning with LoRA...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = fine_tuner.train_epoch(
            train_loader,
            epoch,
            use_wandb=args.use_wandb
        )
        
        val_loss, val_acc = fine_tuner.validate(val_loader, epoch)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss_epoch": train_loss,
                "val_loss": val_loss
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = moondream_config["fine_tuned"]
            fine_tuner.save_model(save_path)
            logger.info(f"✓ New best model saved with val_loss={val_loss:.4f}")
    
    final_save_path = moondream_config["fine_tuned"]
    fine_tuner.save_model(final_save_path)
    
    logger.info("Fine-tuning completed!")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
