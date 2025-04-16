# RoBERTa Fine-Tuned Model for Legal Clause Classification

This directory contains code to fine-tune a RoBERTa model on the CUAD (Contract Understanding Atticus Dataset) for legal clause classification.

## Files

- `train_model.py` - Script to fine-tune RoBERTa on CUAD dataset
- `cuad_preprocessing.py` - Preprocesses CUAD dataset for training
- `infer_clause.py` - Inference script for the trained model
- `risk_map.py` - Maps clause types to risk scores
- `label_map.py` - Mapping between clause types and label IDs
- `fine_tuned_roberta/` - Directory for the trained model files (created during training)

## Usage

### Prerequisites

Before using this code, you'll need to:

1. Install the required Python packages:
   ```
   pip install transformers torch pandas numpy scikit-learn datasets
   ```

2. Get access to the CUAD dataset:
   - This code automatically downloads the CUAD dataset from HuggingFace Datasets
   - You need an internet connection for the initial download

### Training the Model

To train the model:

```bash
# From project root
python -m bert_model.train_model

# With custom parameters
python -m bert_model.train_model --epochs 5 --batch_size 16 --model_name roberta-large
```

**Note:** Training requires a GPU with sufficient memory. On a typical GPU, training can take several hours.

### Using the Trained Model

To run inference on a set of clauses:

```bash
# From project root
python -m bert_model.infer_clause --input_file results/sample_clauses.json
```

This will produce a JSON file with predictions for each clause, including:
- Clause type
- Risk score
- Explanation of the risk

### Comparing with GPT-4o

To compare the model's predictions with GPT-4o:

```bash
python -m bert_model.infer_clause --input_file results/sample_clauses.json --compare_with_gpt results/gpt_predictions.json
```

## Data Requirements

The input file should be a JSON file with one of these formats:

1. A list of clause texts:
   ```json
   [
     "The Contractor shall indemnify...",
     "This Agreement shall remain confidential...",
     ...
   ]
   ```

2. A list of objects with a `clause_text` field:
   ```json
   [
     {"clause_text": "The Contractor shall indemnify..."},
     {"clause_text": "This Agreement shall remain confidential..."},
     ...
   ]
   ```

## Creating Sample Data

You can create a sample dataset from the CUAD test set:

```bash
python -m bert_model.cuad_preprocessing
```

This will generate sample clauses in `results/sample_clauses.json`.