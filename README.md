# AI Legal Triage

AI Legal Triage is a project designed to automatically analyze legal contracts, classify clauses, and assess risk levels. It helps legal professionals prioritize contract review efforts by highlighting potentially risky clauses that require immediate attention.

## Features

- **Clause Extraction**: Extract individual clauses from legal contracts
- **Clause Classification**: Identify the type of legal clause (e.g., Non-Compete, Limitation of Liability)
- **Risk Assessment**: Assign risk scores (0.0-1.0) to clauses based on their type and content
- **Two Model Approaches**:
  - GPT-based classification (high accuracy, external API dependency)
  - Fine-tuned RoBERTa model (local inference, no API dependency)
- **Interactive Web Interface**: Analyze contracts and visualize results in real-time

## Demo

Try the interactive Streamlit app to see the system in action:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/adarench/ai-legal-triage/main/streamlit_apps/roberta_classifier_app.py)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Streamlit

### Installation

1. Clone the repository:
```bash
git clone https://github.com/adarench/AI-Legal-Triage.git
cd AI-Legal-Triage
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_apps/roberta_classifier_app.py
```

## Model Improvements

The project includes two versions of the RoBERTa model:

- **Original Model**: Base fine-tuned model with 60% accuracy
- **Improved Model**: Enhanced model with post-processing rules, achieving 80% accuracy

Key improvements:
- Retrained with more epochs (5 vs 3)
- Lower learning rate (1e-5 vs 2e-5)
- Data augmentation for underrepresented classes
- Post-processing rules for common errors
- Confidence thresholding for uncertain predictions

## Project Structure

- `bert_model/`: RoBERTa model training and inference
- `gpt_mvp/`: GPT-based classification approach
- `streamlit_apps/`: Interactive web applications
- `results/`: Comparison results and sample clauses
- `data/`: Contract datasets and test files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the CUAD (Contract Understanding Atticus Dataset)
- RoBERTa model from HuggingFace Transformers
- OpenAI for GPT API access