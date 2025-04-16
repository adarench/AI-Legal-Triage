# DeepSeek Cloud Training Guide for AI Legal Triage

This guide will walk you through setting up and training your legal clause classification model on DeepSeek Cloud.

## Step 1: Sign Up for DeepSeek Cloud

1. Visit [DeepSeek Cloud](https://www.deepseek.cloud/)
2. Sign up for an account
3. Add payment information (credit card required)

## Step 2: Create a Compute Instance

1. Once logged in, click "Create Instance"
2. Choose your configuration:
   - **Instance Type**: GPU
   - **GPU Type**: A100 (recommended for best performance/price ratio)
   - **Number of GPUs**: 1
   - **Disk Space**: 50 GB (default is sufficient)
   - **Operating System**: Ubuntu 20.04
   - **Framework**: PyTorch (latest)

3. Set a budget limit to prevent unexpected charges:
   - Click "Advanced Options"
   - Set "Auto-shutdown after X hours" to 5 (should be sufficient for training)
   - Enable "Auto-shutdown on idle" with 30 minutes

4. Launch the instance

## Step 3: Upload Training Files

Once your instance is running, you have two options:

### Option A: Direct Git Clone (Recommended)

1. SSH into your instance using the provided connection details
2. Run the deepseek_setup.sh script:
   ```bash
   wget https://raw.githubusercontent.com/adarench/AI-Legal-Triage/main/deepseek_setup.sh
   chmod +x deepseek_setup.sh
   # Edit the script to add your OpenAI API key
   nano deepseek_setup.sh
   # Run the script
   ./deepseek_setup.sh
   ```

### Option B: Upload and Run Optimized Script

1. Connect to your DeepSeek instance via the web terminal or SSH
2. Clone the repository and run the optimized training script:
   ```bash
   git clone https://github.com/adarench/AI-Legal-Triage.git
   cd AI-Legal-Triage
   pip install -r requirements.txt
   
   # Set your OpenAI API key
   export OPENAI_API_KEY="your_key_here"
   
   # Run the DeepSeek-optimized training script
   python -m bert_model.deepseek_train --fp16 --use_trainer
   ```

## Step 4: Monitor Training

Training will take approximately 1-3 hours on an A100 GPU. You can monitor progress:

- Via SSH/terminal: Watch the output logs
- Via DeepSeek web interface: Monitor GPU usage and resource utilization

## Step 5: Download the Trained Model

After training completes:

1. Look for the `fine_tuned_roberta.zip` file in your instance's home directory
2. Download it using the DeepSeek web interface or SCP:
   ```bash
   # From your local machine
   scp username@deepseek-instance-ip:~/fine_tuned_roberta.zip .
   ```

3. Unzip the model locally:
   ```bash
   unzip fine_tuned_roberta.zip -d bert_model/
   ```

## Step 6: Test Your Model

Now you can test your trained model:

```bash
# Navigate to your local AI Legal Triage directory
cd AI-Legal-Triage

# Run inference on sample clauses
python -m bert_model.infer_clause --input_file results/sample_clauses.json
```

## Cost Estimate

Training on DeepSeek Cloud with an A100 GPU:
- A100 pricing: Approximately $1-2 per hour
- Total estimated cost: $3-6 for the entire training process
- Training time: 1-3 hours

## Troubleshooting

- **"CUDA out of memory"**: Reduce batch_size to 16 or 8
- **Training errors**: Check the CUAD dataset loading and preprocessing
- **Connection issues**: Contact DeepSeek support or try reconnecting

## Need Help?

- DeepSeek documentation: [docs.deepseek.cloud](https://docs.deepseek.cloud)
- Contact DeepSeek support via their website
- Check GitHub issues for common problems: [github.com/adarench/AI-Legal-Triage/issues](https://github.com/adarench/AI-Legal-Triage/issues)