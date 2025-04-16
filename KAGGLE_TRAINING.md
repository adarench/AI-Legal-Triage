# Training AI Legal Triage with Kaggle GPU

This guide explains how to use Kaggle's free GPU resources to train the RoBERTa model directly from VS Code.

## Prerequisites

1. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up your Kaggle API credentials:
   - Go to [kaggle.com](https://www.kaggle.com) → Account → API
   - Click "Create New API Token" to download `kaggle.json`
   - Create the directory: `mkdir -p ~/.kaggle`
   - Place this file in `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Step 1: Create a Kaggle Dataset from your project

```bash
# Navigate to the parent directory
cd /Users/adarench/Desktop

# Package your project
zip -r AI-Legal-Triage.zip "AI Legal Triage"

# Create a new Kaggle dataset
kaggle datasets create -p /Users/adarench/Desktop -n ai-legal-triage
```

## Step 2: Create and run a Kaggle kernel

```bash
# Navigate to your project directory
cd /Users/adarench/Desktop/AI\ Legal\ Triage

# Push the kernel to Kaggle and run it
kaggle kernels push

# Check the status of your kernel
kaggle kernels status adarench/ai-legal-triage-training

# Pull the output when complete
kaggle kernels output adarench/ai-legal-triage-training -p /Users/adarench/Desktop
```

## Step 3: Download the trained model

After the kernel completes, you'll find `fine_tuned_roberta.zip` in the kernel output directory.

```bash
# Unzip the model
unzip /Users/adarench/Desktop/fine_tuned_roberta.zip -d /Users/adarench/Desktop/AI\ Legal\ Triage/bert_model/
```

## Troubleshooting

- **Error creating dataset**: Make sure your API token is correctly set up
- **Kernel fails to start**: Check that you have GPU quota available (30 hours/week)
- **Training errors**: You can view logs with `kaggle kernels output`

## Monitoring Training

Monitor your kernel's status and logs:

```bash
# Get the latest logs
kaggle kernels status adarench/ai-legal-triage-training
```

## Time and Resource Usage

- Training should take 3-6 hours on Kaggle's P100 GPU
- You're limited to 30 hours of GPU time per week
- Kaggle provides NVIDIA P100 GPUs with 16GB VRAM