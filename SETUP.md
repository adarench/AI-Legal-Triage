# Legal AI Triage Project Setup Guide

This document provides detailed instructions on how to set up and use the Legal AI Triage system.

## Prerequisites

- Python 3.8+ installed
- GPU with CUDA support (for training the RoBERTa model)
- OpenAI API key (for the GPT-4o based system)

## Installation

1. Clone this repository:
   ```
   git clone [repository-url]
   cd legal-ai-triage
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   ```
   cp .env.template .env
   # Edit .env file and add your OpenAI API key
   ```

## GPT-4o MVP Usage

The GPT-4o MVP is ready to use out of the box. To run the Streamlit app:

```
python run_demo.py app
```

Or directly:

```
streamlit run gpt_mvp/app.py
```

This will launch a web interface where you can:
1. Upload legal contracts in .txt or .docx format
2. Select clauses to analyze
3. View classification results, risk scores, and explanations

## RoBERTa Model Setup

The RoBERTa model requires training on the CUAD dataset before use.

### Dataset

The [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad) is automatically downloaded from HuggingFace during the preprocessing step. It consists of 510 contracts annotated with 41 types of legal clauses.

### Training the Model

To train the RoBERTa model:

```
python run_demo.py train
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size (default: 8)
- `--model_name`: Base model to fine-tune (default: roberta-base)

For example:
```
python run_demo.py train --epochs 5 --batch_size 16 --model_name roberta-large
```

**Note:** Training requires a GPU and may take several hours depending on your hardware.

## Model Comparison

To compare the performance of both models on sample clauses:

```
python run_demo.py compare
```

This will:
1. Run both models on the sample clauses
2. Generate comparison metrics
3. Create visualizations in the results directory

If you want to skip running GPT-4o (to save API costs) and use existing predictions:
```
python run_demo.py compare --skip_gpt
```

## Project Structure

```
legal-risk-mvp/
├── gpt_mvp/                     # GPT-4 powered MVP
│   ├── app.py                   # Streamlit UI
│   ├── gpt_clauses.py           # OpenAI API logic
│   ├── clause_parser.py         # Text parsing logic
│   └── prompt_template.py       # Prompt definition
│
├── bert_model/                  # Fine-tuned model demo
│   ├── train_model.py           # Training script
│   ├── cuad_preprocessing.py    # CUAD preprocessing
│   ├── infer_clause.py          # Inference script
│   ├── risk_map.py              # Clause to risk mapping
│   ├── label_map.py             # Label encodings
│   └── fine_tuned_roberta/      # Trained model (created during training)
│
├── results/                     # Output files
│   ├── sample_clauses.json      # Test clauses
│   ├── model_comparison.py      # Comparison script
│   ├── *_predictions.json       # Generated model predictions
│   └── *.png                    # Generated visualizations
│
├── data/                        # Data directory
│   ├── cuad_processed/          # Processed CUAD dataset (created during training)
│   └── external/                # Any external data
│
├── run_demo.py                  # Main CLI script
├── requirements.txt             # Dependencies
└── .env                         # Environment variables
```

## Dependencies

The main dependencies are:
- `streamlit`: For the web interface
- `openai`: For GPT-4o API access
- `transformers`, `torch`: For RoBERTa model
- `pandas`, `numpy`: For data manipulation
- `matplotlib`, `seaborn`: For visualizations
- `python-docx`: For parsing Word documents

## Notes and Limitations

1. **GPT-4o API Costs**: Using the GPT-4o API incurs costs based on the number of tokens processed. Monitor your usage to avoid unexpected charges.

2. **GPU Requirements**: Training the RoBERTa model requires a GPU with at least 8GB of memory. For optimal performance, 16GB+ is recommended.

3. **Clause Parsing**: The current clause parsing logic is heuristic-based and may not correctly identify all clauses in complex contracts. Results may vary based on document formatting.

4. **Risk Scoring**: The risk scores are based on predefined mappings and should be reviewed by legal experts for your specific use case.

## Extending the Project

To extend this project:

1. **Add More Clause Types**: Update `bert_model/label_map.py` and `bert_model/risk_map.py` to include additional clause types.

2. **Improve Clause Parsing**: Enhance `gpt_mvp/clause_parser.py` with more sophisticated parsing logic (e.g., NLP-based segmentation).

3. **Custom Risk Scoring**: Modify risk assessment logic in both models based on your organization's specific risk profile.

4. **Integration with Legal Databases**: Add features to reference legal precedents or regulatory requirements related to identified clauses.