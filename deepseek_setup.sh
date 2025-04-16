#!/bin/bash
# DeepSeek Cloud Setup Script for AI Legal Triage

# Clone the repository
git clone https://github.com/adarench/AI-Legal-Triage.git
cd AI-Legal-Triage

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"  # Replace with your API key

# Make directory for CUAD processed data
mkdir -p data/cuad_processed
mkdir -p bert_model/fine_tuned_roberta

# Run training with optimal parameters for A100 GPU
python -m bert_model.train_model \
  --epochs 3 \
  --batch_size 32 \
  --model_name roberta-base \
  --learning_rate 3e-5 \
  --max_length 512

# Compress the trained model for easy download
cd ..
zip -r fine_tuned_roberta.zip AI-Legal-Triage/bert_model/fine_tuned_roberta/

echo "====================================="
echo "Training complete! Download the model from the files tab."
echo "====================================="