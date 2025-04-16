#!/usr/bin/env python
"""
Training script for Kaggle GPU.
Run this with the Kaggle API to train on their GPU.
"""

import os
import sys
import subprocess

# Display available resources
print("Available GPU information:")
subprocess.run(["nvidia-smi"])

# Unzip and set up project
if not os.path.exists("AI-Legal-Triage"):
    print("Setting up project...")
    os.makedirs("AI-Legal-Triage", exist_ok=True)
    
    # Unzip the project (if using as a dataset)
    if os.path.exists("/kaggle/input/ai-legal-triage/AI-Legal-Triage.zip"):
        subprocess.run(["unzip", "/kaggle/input/ai-legal-triage/AI-Legal-Triage.zip", "-d", "."])
    else:
        # Clone from GitHub as fallback
        subprocess.run(["git", "clone", "https://github.com/adarench/AI-Legal-Triage.git"])

# Navigate to project directory
os.chdir("AI-Legal-Triage")

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Create necessary directories
os.makedirs("data/cuad_processed", exist_ok=True)
os.makedirs("bert_model/fine_tuned_roberta", exist_ok=True)

# Run the training
print("Starting training...")
subprocess.run([
    sys.executable, "-m", "bert_model.train_model",
    "--epochs", "3",
    "--batch_size", "16",  # P100 can handle this batch size
    "--learning_rate", "3e-5",
    "--max_length", "512"
])

# Compress the model for download
print("Compressing model for download...")
subprocess.run(["zip", "-r", "fine_tuned_roberta.zip", "bert_model/fine_tuned_roberta/"])

print("Training complete! Download the fine_tuned_roberta.zip file.")