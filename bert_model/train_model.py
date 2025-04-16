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
    get_linear_schedule_with_warmup
)

from .label_map import LABEL_MAP, REVERSE_LABEL_MAP
from .cuad_preprocessing import CUADPreprocessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LegalClauseDataset(Dataset):
    """Dataset for legal clause classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset.
        
        Args:
            texts: List of clause texts
            labels: List of label IDs
            tokenizer: Huggingface tokenizer
            max_length: Maximum sequence length
        """
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
    batch_size=8,
    epochs=3,
    learning_rate=2e-5,
    model_name="roberta-base",
    max_length=512,
    seed=42
):
    """
    Train a RoBERTa model on the CUAD dataset.
    
    Args:
        data_dir: Directory with processed CUAD data
        output_dir: Directory to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        model_name: Base model name
        max_length: Max sequence length
        seed: Random seed
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    except FileNotFoundError:
        print("Data files not found. Running preprocessing...")
        preprocessor = CUADPreprocessor(output_dir=data_dir)
        train_df, val_df, test_df = preprocessor.download_and_preprocess()
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP),
        problem_type="single_label_classification"
    )
    model.to(device)
    
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0.0
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
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
    
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa model on CUAD dataset")
    parser.add_argument("--data_dir", default="data/cuad_processed", help="Directory with processed data")
    parser.add_argument("--output_dir", default="bert_model/fine_tuned_roberta", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", default="roberta-base", help="Base model name")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
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
        seed=args.seed
    )

# Notes:
# 1. This script requires a GPU for efficient training (using CUDA)
# 2. You'll need to first run the cuad_preprocessing.py script to prepare the data
# 3. Training with the default parameters will take several hours on a decent GPU
# 4. For a quick test, reduce epochs to 1 and max_length to 128
# 5. You may need to adjust batch_size based on your GPU memory