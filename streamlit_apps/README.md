# AI Legal Triage Streamlit Apps

This directory contains Streamlit web applications for the AI Legal Triage project.

## Available Apps

- **roberta_classifier_app.py**: Interactive app for analyzing legal clauses with the fine-tuned RoBERTa model. Compares the original and improved models to showcase enhancements.

## Model Comparison Features

The roberta_classifier_app.py showcases:

1. **Side-by-side Model Comparison**
   - Compare original and improved model predictions on the same clause
   - See confidence scores and risk assessments from both models
   - Observe post-processing rule application in action

2. **Performance Metrics**
   - Visual comparison of accuracy improvements
   - Confidence score increases across categories
   - Target category enhancements

3. **Interactive Testing**
   - Input custom clauses
   - Select from sample clauses
   - Upload clause files

## Running Locally

```bash
# From project root
streamlit run streamlit_apps/roberta_classifier_app.py
```

## Deployment

The app is deployed on Streamlit Cloud at:
https://ai-legal-triage.streamlit.app/