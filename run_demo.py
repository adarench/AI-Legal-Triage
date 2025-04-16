#!/usr/bin/env python
"""
Demo script for Legal AI Triage system.

This script provides a command-line interface to:
1. Run the GPT-4o Streamlit app
2. Train the RoBERTa model
3. Compare model performance
"""

import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_openai_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key by:")
        print("1. Creating a .env file with OPENAI_API_KEY=your_key_here")
        print("2. Or set it in your environment with export OPENAI_API_KEY=your_key_here")
        return False
    return True

def run_streamlit_app():
    """Run the Streamlit app."""
    if not check_openai_key():
        return
    
    print("Starting Streamlit app for GPT-4o legal clause analysis...")
    subprocess.run(["streamlit", "run", "gpt_mvp/app.py"])

def train_roberta_model(args):
    """Train the RoBERTa model."""
    print("Starting RoBERTa model training...")
    print("WARNING: This requires a GPU and may take several hours.")
    
    train_args = [
        "python", "-m", "bert_model.train_model",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ]
    
    if args.model_name:
        train_args.extend(["--model_name", args.model_name])
    
    try:
        subprocess.run(train_args)
    except Exception as e:
        print(f"Error training model: {str(e)}")

def run_comparison(args):
    """Run model comparison."""
    if not check_openai_key() and not args.skip_gpt:
        return
    
    print("Running model comparison...")
    
    comparison_args = [
        "python", "results/model_comparison.py",
        "--sample_file", args.sample_file
    ]
    
    if args.skip_gpt:
        comparison_args.append("--skip_gpt")
    
    try:
        subprocess.run(comparison_args)
    except Exception as e:
        print(f"Error running comparison: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Legal AI Triage Demo")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Streamlit app
    parser_app = subparsers.add_parser("app", help="Run the Streamlit app")
    
    # Train model
    parser_train = subparsers.add_parser("train", help="Train the RoBERTa model")
    parser_train.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser_train.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser_train.add_argument("--model_name", help="Base model name (e.g., roberta-base)")
    
    # Compare models
    parser_compare = subparsers.add_parser("compare", help="Compare GPT and RoBERTa models")
    parser_compare.add_argument("--sample_file", default="results/sample_clauses.json", 
                               help="Path to sample clauses JSON file")
    parser_compare.add_argument("--skip_gpt", action="store_true", 
                               help="Skip running GPT and use existing predictions")
    
    args = parser.parse_args()
    
    if args.command == "app":
        run_streamlit_app()
    elif args.command == "train":
        train_roberta_model(args)
    elif args.command == "compare":
        run_comparison(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()