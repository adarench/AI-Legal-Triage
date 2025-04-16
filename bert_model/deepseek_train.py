"""
Optimized training script for DeepSeek Cloud A100 GPUs.
This script includes mixed precision training and checkpointing.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
import time
from pathlib import Path

# Fix imports whether run as module or script
try:
    from .label_map import LABEL_MAP, REVERSE_LABEL_MAP
    from .cuad_preprocessing import CUADPreprocessor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bert_model.label_map import LABEL_MAP, REVERSE_LABEL_MAP
    from bert_model.cuad_preprocessing import CUADPreprocessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LegalClauseDataset(Dataset):
    """Dataset for legal clause classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train_model(
    data_dir="data/cuad_processed",
    output_dir="bert_model/fine_tuned_roberta",
    batch_size=32,  # Increased for A100
    epochs=3,
    learning_rate=3e-5,
    model_name="roberta-base",
    max_length=512,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    use_trainer=True,  # Use HF Trainer for better performance
    seed=42
):
    """
    Train a RoBERTa model on the CUAD dataset with optimizations for DeepSeek Cloud.
    """
    start_time = time.time()
    print(f"🚀 Starting optimized training on {device}")
    print(f"🖥️  Using {'mixed precision (fp16)' if fp16 else 'full precision'}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create processed data
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        print("📊 Loaded preprocessed data")
    except FileNotFoundError:
        print("🔍 Preprocessed data not found. Running preprocessing...")
        preprocessor = CUADPreprocessor(output_dir=data_dir)
        train_df, val_df, test_df = preprocessor.download_and_preprocess()
        print("✅ Preprocessing complete")
    
    print(f"📝 Training samples: {len(train_df)}")
    print(f"📝 Validation samples: {len(val_df)}")
    print(f"📝 Test samples: {len(test_df)}")
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        problem_type="single_label_classification"
    )
    model.to(device)
    print(f"🤖 Loaded model: {model_name}")
    
    # Create datasets
    train_dataset = LegalClauseDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = LegalClauseDataset(
        val_df["text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_length
    )
    
    test_dataset = LegalClauseDataset(
        test_df["text"].tolist(),
        test_df["label_id"].tolist(),
        tokenizer,
        max_length
    )
    
    if use_trainer:
        # Use HuggingFace Trainer for optimized training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_total_limit=2,
            fp16=fp16,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",  # Disable wandb, etc.
        )
        
        # Define compute_metrics function for the trainer
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            f1 = f1_score(labels, preds, average="weighted")
            acc = accuracy_score(labels, preds)
            return {
                "accuracy": acc,
                "f1": f1,
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        print("🏋️ Starting training with HuggingFace Trainer...")
        trainer.train()
        
        # Save best model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Evaluate
        print("📊 Evaluating on test set...")
        results = trainer.evaluate(test_dataset)
        print(f"Test results: {results}")
        
        # Save test metrics
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
    else:
        # Manual training loop (alternative)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Use mixed precision if fp16 is enabled
        scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Training loop
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                if fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()
                
                if step % logging_steps == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                if step % save_steps == 0 and step > 0:
                    # Save checkpoint
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}-{step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"Saved checkpoint to {checkpoint_dir}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    true = labels.cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_true.extend(true)
            
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(val_true, val_preds, average="weighted")
            val_acc = accuracy_score(val_true, val_preds)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Saved best model with Val F1: {val_f1:.4f}")
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                true = labels.cpu().numpy()
                
                test_preds.extend(preds)
                test_true.extend(true)
        
        # Calculate and save metrics
        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average="weighted")
        
        test_report = classification_report(
            test_true, 
            test_preds, 
            target_names=list(LABEL_MAP.keys()),
            output_dict=True
        )
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        # Save test metrics
        metrics = {
            "accuracy": test_acc,
            "f1_score": test_f1,
            "classification_report": test_report
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"🎉 Training complete! Model saved to {output_dir}")
    print(f"⏱️ Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Create a model card
    model_card = f"""
# Legal Clause Classification Model

This model is fine-tuned from {model_name} on the CUAD (Contract Understanding Atticus Dataset) for legal clause classification.

## Model details
- Base model: {model_name}
- Number of labels: {len(LABEL_MAP)}
- Training dataset: CUAD
- Training time: {int(hours)}h {int(minutes)}m
- GPU used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

## Performance
- F1 Score: {test_f1:.4f}
- Accuracy: {test_acc:.4f}

## Usage
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("path/to/model")
model = RobertaForSequenceClassification.from_pretrained("path/to/model")

# Classify a legal clause
text = "The Contractor shall indemnify and hold harmless the Company..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
```

## License
This model uses the CUAD dataset, which has its own licensing terms. Please refer to the original dataset for more information.

## Created with DeepSeek Cloud
    """
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa model on CUAD dataset (DeepSeek Optimized)")
    parser.add_argument("--data_dir", default="data/cuad_processed", help="Directory with processed data")
    parser.add_argument("--output_dir", default="bert_model/fine_tuned_roberta", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--model_name", default="roberta-base", help="Base model name")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (faster)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--use_trainer", action="store_true", help="Use HuggingFace Trainer API")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        max_length=args.max_length,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_trainer=args.use_trainer,
        seed=args.seed
    )