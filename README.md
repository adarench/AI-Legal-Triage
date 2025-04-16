# Legal Contract Clause Risk Analyzer

This project is an MVP for a legal contract analysis system that uses AI to identify, classify, and assess risk levels in legal clauses.

## Features

- Parse legal contracts (.txt or .docx files) and extract individual clauses
- Classify clauses by type (e.g., Indemnification, Confidentiality)
- Assign risk scores from 0.0 to 1.0
- Provide plain-language explanations of potential risks
- Visualize results and download analysis as JSON

## Two Implementation Tracks

1. **GPT-4o Powered MVP** - Production-ready Streamlit application using OpenAI's GPT-4o API for sophisticated legal analysis
2. **RoBERTa Fine-Tuned Model** - Technical demo using a fine-tuned RoBERTa model on the CUAD dataset

## Getting Started

### Prerequisites

```
# Install dependencies
pip install -r requirements.txt
```

### GPT-4o MVP

```
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Launch the Streamlit app
cd gpt_mvp
streamlit run app.py
```

### Fine-Tuned Model

```
# Train the model (warning: requires GPU)
cd bert_model
python train_model.py

# Run inference on clauses
python infer_clause.py
```

## Project Structure

```
legal-risk-mvp/
├── gpt_mvp/                     # GPT-4 powered MVP
│   ├── app.py                   # Streamlit UI
│   ├── gpt_clauses.py           # OpenAI API logic
│   ├── clause_parser.py         # Text parsing logic
│   ├── prompt_template.py       # Prompt definition
│   └── __init__.py
│
├── bert_model/                  # Fine-tuned model demo
│   ├── train_model.py           # Training script
│   ├── cuad_preprocessing.py    # CUAD preprocessing
│   ├── infer_clause.py          # Inference script
│   ├── risk_map.py              # Clause to risk mapping
│   ├── label_map.py             # Label encodings
│   ├── fine_tuned_roberta/      # Trained model
│   └── __init__.py
│
├── results/                     # Output files
│   ├── sample_clauses.json      # Test clauses
│   ├── model_comparison.json    # Comparison results
│   └── comparison_table.csv     # Summary CSV
│
└── data/                        # Data directory
    ├── test_contracts/          # Test contracts
    └── external/                # External data
```
