"""
Script to compare GPT-4o and RoBERTa model predictions.

This script:
1. Loads sample clauses
2. Runs both models on the same clauses
3. Compares the results and generates metrics
4. Creates visualization and tables
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpt_mvp.gpt_clauses import GPTClauseAnalyzer
from bert_model.infer_clause import RobertaClausePredictor
from bert_model.infer_clause import compare_with_gpt


def run_comparison(
    sample_file="results/sample_clauses.json",
    gpt_output_file="results/gpt_predictions.json",
    roberta_output_file="results/roberta_predictions.json",
    comparison_output_file="results/model_comparison.json",
    csv_output_file="results/comparison_table.csv",
    api_key=None
):
    """Run both models and compare their predictions."""
    
    # Load sample clauses
    with open(sample_file, "r") as f:
        samples = json.load(f)
    
    # Extract clauses
    if all(isinstance(item, dict) and "clause_text" in item for item in samples):
        clauses = [item["clause_text"] for item in samples]
    else:
        clauses = samples
    
    # Check if we have GPT API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            print("No OpenAI API key provided. Skipping GPT analysis.")
            return None
    
    # Run GPT model
    print("Running GPT-4o analysis...")
    gpt_analyzer = GPTClauseAnalyzer(api_key=api_key)
    gpt_results = gpt_analyzer.batch_analyze(clauses)
    
    # Save GPT results
    gpt_output = []
    for i, clause in enumerate(clauses):
        gpt_output.append({
            "clause_text": clause,
            **gpt_results[i]
        })
    
    with open(gpt_output_file, "w") as f:
        json.dump(gpt_output, f, indent=2)
    
    print(f"GPT predictions saved to {gpt_output_file}")
    
    # Try to run RoBERTa model if available
    try:
        print("Running RoBERTa analysis...")
        roberta_predictor = RobertaClausePredictor()
        roberta_results = roberta_predictor.batch_predict(clauses)
        
        # Save RoBERTa results
        roberta_output = []
        for i, clause in enumerate(clauses):
            roberta_output.append({
                "clause_text": clause,
                **roberta_results[i]
            })
        
        with open(roberta_output_file, "w") as f:
            json.dump(roberta_output, f, indent=2)
        
        print(f"RoBERTa predictions saved to {roberta_output_file}")
        
        # Compare results
        comparison = compare_with_gpt(
            roberta_output, 
            gpt_output,
            comparison_output_file
        )
        
        # Create CSV for easy comparison
        csv_data = []
        for item in comparison["comparisons"]:
            csv_data.append({
                "clause_excerpt": item["clause_text"][:100] + "..." if len(item["clause_text"]) > 100 else item["clause_text"],
                "GPT_type": item["gpt"]["type"],
                "RoBERTa_type": item["roberta"]["type"],
                "GPT_risk": item["gpt"]["risk_score"],
                "RoBERTa_risk": item["roberta"]["risk_score"],
                "Risk_diff": abs(item["gpt"]["risk_score"] - item["roberta"]["risk_score"])
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_file, index=False)
        print(f"Comparison table saved to {csv_output_file}")
        
        return comparison
    
    except Exception as e:
        print(f"RoBERTa model not available: {str(e)}")
        print("Proceeding with GPT results only.")
        return None


def visualize_comparison(
    comparison_file="results/model_comparison.json",
    output_dir="results"
):
    """Create visualizations of the model comparison."""
    # Load comparison results
    with open(comparison_file, "r") as f:
        comparison = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    clause_types = []
    gpt_scores = []
    roberta_scores = []
    
    for item in comparison["comparisons"]:
        clause_types.append(item["gpt"]["type"])
        gpt_scores.append(item["gpt"]["risk_score"])
        roberta_scores.append(item["roberta"]["risk_score"])
    
    # Plot risk score comparison
    plt.figure(figsize=(10, 6))
    x = range(len(clause_types))
    plt.bar([i-0.2 for i in x], gpt_scores, width=0.4, label="GPT-4o", color="blue")
    plt.bar([i+0.2 for i in x], roberta_scores, width=0.4, label="RoBERTa", color="red")
    plt.xticks(x, clause_types, rotation=45, ha="right")
    plt.xlabel("Clause Type")
    plt.ylabel("Risk Score")
    plt.title("Risk Score Comparison: GPT-4o vs RoBERTa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_score_comparison.png"))
    
    # Plot agreement heatmap
    df = pd.DataFrame({
        "GPT Type": [item["gpt"]["type"] for item in comparison["comparisons"]],
        "RoBERTa Type": [item["roberta"]["type"] for item in comparison["comparisons"]],
    })
    
    agreement_count = pd.crosstab(df["GPT Type"], df["RoBERTa Type"])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(agreement_count, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Agreement Matrix: GPT-4o vs RoBERTa")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agreement_matrix.png"))
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare GPT-4o and RoBERTa predictions")
    parser.add_argument("--sample_file", default="results/sample_clauses.json", 
                        help="Path to sample clauses JSON file")
    parser.add_argument("--api_key", 
                        help="OpenAI API key (can also be set as OPENAI_API_KEY environment variable)")
    parser.add_argument("--skip_gpt", action="store_true", 
                        help="Skip running GPT and use existing predictions")
    
    args = parser.parse_args()
    
    if not args.skip_gpt:
        comparison = run_comparison(sample_file=args.sample_file, api_key=args.api_key)
    
    # Try to visualize if comparison results exist
    comparison_file = "results/model_comparison.json"
    if os.path.exists(comparison_file):
        try:
            visualize_comparison(comparison_file)
        except Exception as e:
            print(f"Failed to create visualizations: {str(e)}")
    else:
        print(f"Comparison file {comparison_file} not found. Skipping visualization.")


if __name__ == "__main__":
    main()

# Notes:
# 1. You need to set your OpenAI API key to run the GPT model
# 2. You need to have a fine-tuned RoBERTa model to run the comparison
# 3. This script will create visualizations to compare the models
# 4. Example usage:
#    python results/model_comparison.py
# 5. If you don't want to run GPT again (to save API costs):
#    python results/model_comparison.py --skip_gpt