import os
import json
import argparse
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

from .label_map import REVERSE_LABEL_MAP
from .risk_map import get_risk_score, get_risk_explanation

class RobertaClausePredictor:
    """Class for making predictions with a fine-tuned RoBERTa model."""
    
    def __init__(self, model_dir="bert_model/fine_tuned_roberta"):
        """
        Initialize the predictor with a fine-tuned model.
        
        Args:
            model_dir: Directory with the fine-tuned model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {model_dir}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_dir}: {str(e)}")
    
    def predict_clause(self, clause_text, include_explanation=True):
        """
        Predict the type and risk score for a legal clause.
        
        Args:
            clause_text: The text of the legal clause
            include_explanation: Whether to include risk explanation
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize input
        inputs = self.tokenizer(
            clause_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get predicted class and its probability
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
        
        # Get class label
        predicted_type = REVERSE_LABEL_MAP.get(pred_class, "Unknown")
        
        # Get risk score
        risk_score = get_risk_score(predicted_type)
        
        # Prepare result
        result = {
            "type": predicted_type,
            "risk_score": risk_score,
            "model_confidence": confidence
        }
        
        # Add explanation if requested
        if include_explanation:
            result["explanation"] = get_risk_explanation(predicted_type, risk_score)
        
        return result
    
    def batch_predict(self, clauses, include_explanation=True):
        """
        Predict types and risk scores for multiple clauses.
        
        Args:
            clauses: List of clause texts
            include_explanation: Whether to include risk explanations
            
        Returns:
            List of dictionaries with prediction results
        """
        return [self.predict_clause(clause, include_explanation) for clause in clauses]
    
    def predict_from_file(self, input_file, output_file=None, include_explanation=True):
        """
        Predict from a JSON file containing clauses.
        
        Args:
            input_file: Path to JSON file with clause texts
            output_file: Path to save results (if None, uses input_file with _predictions suffix)
            include_explanation: Whether to include risk explanations
            
        Returns:
            List of dictionaries with prediction results
        """
        # Load clauses from file
        with open(input_file, "r") as f:
            data = json.load(f)
        
        # Extract clause texts
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                clauses = data
            elif all(isinstance(item, dict) and "clause_text" in item for item in data):
                clauses = [item["clause_text"] for item in data]
            else:
                raise ValueError("Input JSON must be a list of strings or dictionaries with 'clause_text' field")
        else:
            raise ValueError("Input JSON must be a list")
        
        # Make predictions
        results = self.batch_predict(clauses, include_explanation)
        
        # If input was list of dicts, merge results back into original data
        if all(isinstance(item, dict) for item in data):
            for i, item in enumerate(data):
                item.update(results[i])
            output_data = data
        else:
            # Create result dictionaries
            output_data = []
            for i, clause in enumerate(clauses):
                output_data.append({
                    "clause_text": clause,
                    **results[i]
                })
        
        # Save results if output file is specified
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_predictions{ext}"
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Predictions saved to {output_file}")
        return output_data


def compare_with_gpt(
    roberta_predictions,
    gpt_predictions,
    output_file="results/model_comparison.json"
):
    """
    Compare RoBERTa predictions with GPT predictions.
    
    Args:
        roberta_predictions: Path to RoBERTa predictions or prediction objects
        gpt_predictions: Path to GPT predictions or prediction objects
        output_file: Path to save comparison results
    """
    # Load predictions if file paths were provided
    if isinstance(roberta_predictions, str):
        with open(roberta_predictions, "r") as f:
            roberta_predictions = json.load(f)
    
    if isinstance(gpt_predictions, str):
        with open(gpt_predictions, "r") as f:
            gpt_predictions = json.load(f)
    
    # Check if the number of predictions match
    if len(roberta_predictions) != len(gpt_predictions):
        print("Warning: Number of predictions doesn't match")
        # Try to match predictions by clause text
        matched_predictions = []
        for r_pred in roberta_predictions:
            r_text = r_pred.get("clause_text", "")
            for g_pred in gpt_predictions:
                g_text = g_pred.get("clause_text", "")
                if r_text == g_text:
                    matched_predictions.append({
                        "clause_text": r_text,
                        "roberta": {
                            "type": r_pred.get("type", ""),
                            "risk_score": r_pred.get("risk_score", 0.0),
                            "explanation": r_pred.get("explanation", "")
                        },
                        "gpt": {
                            "type": g_pred.get("type", ""),
                            "risk_score": g_pred.get("risk_score", 0.0),
                            "explanation": g_pred.get("explanation", "")
                        }
                    })
                    break
        
        comparison = matched_predictions
    else:
        # Combine predictions
        comparison = []
        for i in range(len(roberta_predictions)):
            r_pred = roberta_predictions[i]
            g_pred = gpt_predictions[i]
            
            comparison.append({
                "clause_text": r_pred.get("clause_text", g_pred.get("clause_text", "")),
                "roberta": {
                    "type": r_pred.get("type", ""),
                    "risk_score": r_pred.get("risk_score", 0.0),
                    "explanation": r_pred.get("explanation", "")
                },
                "gpt": {
                    "type": g_pred.get("type", ""),
                    "risk_score": g_pred.get("risk_score", 0.0),
                    "explanation": g_pred.get("explanation", "")
                }
            })
    
    # Calculate agreement metrics
    type_agreements = 0
    risk_score_diffs = []
    
    for item in comparison:
        # Check if types match
        if item["roberta"]["type"] == item["gpt"]["type"]:
            type_agreements += 1
        
        # Calculate risk score difference
        risk_diff = abs(item["roberta"]["risk_score"] - item["gpt"]["risk_score"])
        risk_score_diffs.append(risk_diff)
    
    # Calculate metrics
    type_agreement_pct = (type_agreements / len(comparison)) * 100 if comparison else 0
    avg_risk_diff = np.mean(risk_score_diffs) if risk_score_diffs else 0
    
    # Add summary metrics
    summary = {
        "num_clauses": len(comparison),
        "type_agreement_percent": type_agreement_pct,
        "avg_risk_score_difference": avg_risk_diff
    }
    
    output = {
        "summary": summary,
        "comparisons": comparison
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Comparison saved to {output_file}")
    print(f"Type agreement: {type_agreement_pct:.2f}%")
    print(f"Average risk score difference: {avg_risk_diff:.4f}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Predict legal clause types and risk scores")
    parser.add_argument("--model_dir", default="bert_model/fine_tuned_roberta", 
                        help="Directory with fine-tuned model")
    parser.add_argument("--input_file", required=True, 
                        help="JSON file with clauses to analyze")
    parser.add_argument("--output_file", 
                        help="File to save prediction results (default: input_file_predictions.json)")
    parser.add_argument("--no_explanation", action="store_true", 
                        help="Don't include risk explanations in output")
    parser.add_argument("--compare_with_gpt", 
                        help="Path to GPT predictions to compare with")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RobertaClausePredictor(model_dir=args.model_dir)
    
    # Make predictions
    predictions = predictor.predict_from_file(
        args.input_file,
        args.output_file,
        not args.no_explanation
    )
    
    # Compare with GPT if specified
    if args.compare_with_gpt:
        compare_with_gpt(predictions, args.compare_with_gpt)


if __name__ == "__main__":
    main()

# Notes:
# 1. You need to have a fine-tuned model in the specified directory
# 2. This script can be used to analyze clauses from a JSON file
# 3. It can also compare RoBERTa predictions with GPT predictions
# 4. Example usage:
#    python -m bert_model.infer_clause --input_file results/sample_clauses.json
# 5. If you want to compare with GPT:
#    python -m bert_model.infer_clause --input_file results/sample_clauses.json --compare_with_gpt results/gpt_predictions.json