#!/usr/bin/env python
"""
This script preprocesses the CUAD dataset for the RoBERTa model.
Run this script to download and prepare the CUAD dataset.
"""

import os
import sys
import importlib.util

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

# Dynamically import the cuad_preprocessing module
spec = importlib.util.spec_from_file_location(
    "cuad_preprocessing", 
    os.path.join("bert_model", "cuad_preprocessing.py")
)
cuad_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuad_preprocessing)

# Import label_map directly
from bert_model.label_map import LABEL_MAP

def main():
    print("Starting CUAD dataset preprocessing...")
    
    # Create output directory
    output_dir = "data/cuad_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = cuad_preprocessing.CUADPreprocessor(output_dir=output_dir)
    
    try:
        # Download and preprocess dataset
        print("Downloading and preprocessing CUAD dataset...")
        train_df, val_df, test_df = preprocessor.download_and_preprocess()
        
        # Create sample file
        print("Creating sample clauses file...")
        preprocessor.create_sample_file(filename="results/cuad_samples.json")
        
        print(f"Preprocessing complete! Data saved to {output_dir}")
        print(f"Number of training examples: {len(train_df)}")
        print(f"Number of validation examples: {len(val_df)}")
        print(f"Number of testing examples: {len(test_df)}")
        print(f"Number of labels: {len(LABEL_MAP)}")
        
        # Show label distribution
        label_counts = train_df.label.value_counts()
        print("\nLabel distribution in training set:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
    except Exception as e:
        print(f"Error preprocessing CUAD dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())