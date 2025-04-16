import os
import json
import pandas as pd
from datasets import load_dataset
from .label_map import LABEL_MAP

class CUADPreprocessor:
    """Class to preprocess the CUAD dataset for fine-tuning RoBERTa."""
    
    def __init__(self, output_dir="data/cuad_processed"):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_and_preprocess(self, save_to_disk=True):
        """
        Download the CUAD dataset and preprocess it for training.
        
        Args:
            save_to_disk: Whether to save processed data to disk
        
        Returns:
            train_df, val_df, test_df: Pandas DataFrames for train/val/test splits
        """
        print("Downloading CUAD dataset from HuggingFace...")
        # CUAD dataset is available on HuggingFace
        # NOTE: You'll need to be connected to the internet to download this
        cuad_dataset = load_dataset("cuad")
        
        print("Processing dataset...")
        # Extract relevant fields and format data for single-label classification
        train_data = self._process_split(cuad_dataset["train"])
        val_data = self._process_split(cuad_dataset["validation"])
        test_data = self._process_split(cuad_dataset["test"])
        
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)
        
        if save_to_disk:
            train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)
            test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)
            
            # Save metadata
            label_counts = train_df.label.value_counts().to_dict()
            metadata = {
                "num_train_examples": len(train_df),
                "num_val_examples": len(val_df),
                "num_test_examples": len(test_df),
                "num_labels": len(LABEL_MAP),
                "label_counts": label_counts
            }
            
            with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Preprocessing complete. Saved to {self.output_dir}")
        return train_df, val_df, test_df
    
    def _process_split(self, dataset_split):
        """
        Process a split of the CUAD dataset.
        
        Args:
            dataset_split: A split from the CUAD dataset
            
        Returns:
            processed_data: List of dictionaries with processed examples
        """
        processed_data = []
        
        for example in dataset_split:
            # Each example has contract text and multiple labels with answers
            contract_text = example["contract_text"]
            
            # For each extracted clause (answer), create a single-label example
            for i, answer in enumerate(example["answers"]):
                if not answer["text"]:  # Skip empty answers
                    continue
                
                # Get the label (category) for this clause
                label = example["categories"][i]
                if label not in LABEL_MAP:
                    label = "Uncategorized"
                
                processed_data.append({
                    "text": answer["text"],
                    "label": label,
                    "label_id": LABEL_MAP[label],
                    "contract_id": example["contract_id"]
                })
        
        return processed_data
    
    def create_sample_file(self, filename="results/sample_clauses.json", num_samples=5):
        """
        Create a sample file with examples from each category for testing.
        
        Args:
            filename: Output filename
            num_samples: Number of samples per category
        """
        try:
            test_df = pd.read_csv(os.path.join(self.output_dir, "test.csv"))
        except FileNotFoundError:
            print("Test file not found. Run download_and_preprocess() first.")
            return
        
        samples = []
        
        # Get samples from each category
        for label in LABEL_MAP.keys():
            label_examples = test_df[test_df.label == label]
            if len(label_examples) > 0:
                category_samples = label_examples.sample(min(num_samples, len(label_examples)))
                
                for _, row in category_samples.iterrows():
                    samples.append({
                        "clause_text": row["text"],
                        "type": row["label"]
                    })
        
        # Save to file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(samples, f, indent=2)
            
        print(f"Created sample file with {len(samples)} examples at {filename}")


# Example usage (commented out)
"""
if __name__ == "__main__":
    preprocessor = CUADPreprocessor()
    train_df, val_df, test_df = preprocessor.download_and_preprocess()
    preprocessor.create_sample_file()
"""

# NOTE: You need to download the CUAD dataset to proceed.
# The dataset will be downloaded automatically from HuggingFace 
# when you run the download_and_preprocess method.
# This requires an internet connection.