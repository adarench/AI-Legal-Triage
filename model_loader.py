"""
Model loading utilities for the Streamlit app.

This module facilitates loading models from either local storage or Hugging Face Hub.
"""

import os
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Default repository information
DEFAULT_MODEL_REPO = "adarench/legal-roberta-classifier"  # You'll need to create this
DEFAULT_ORIGINAL_MODEL_PATH = "original"
DEFAULT_IMPROVED_MODEL_PATH = "improved" 

def load_model_from_hf_hub(model_path, use_auth_token=None):
    """
    Load a model from Hugging Face Hub
    """
    if not HF_HUB_AVAILABLE:
        st.error("Hugging Face Hub is not available. Please install with 'pip install huggingface-hub'")
        return None
    
    try:
        # The repository should contain all tokenizer and model files
        tokenizer = RobertaTokenizer.from_pretrained(model_path, use_auth_token=use_auth_token)
        model = RobertaForSequenceClassification.from_pretrained(model_path, use_auth_token=use_auth_token)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {str(e)}")
        return None, None

def load_model_local(model_dir):
    """
    Load a model from local directory
    """
    try:
        if not os.path.exists(model_dir):
            st.error(f"Model directory {model_dir} does not exist")
            return None, None
        
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        model = RobertaForSequenceClassification.from_pretrained(model_dir)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from local path: {str(e)}")
        return None, None

def load_model(model_type="improved", use_hf_hub=False, local_path=None, hf_token=None):
    """
    Load a model either locally or from Hugging Face Hub
    
    Args:
        model_type: 'original' or 'improved'
        use_hf_hub: Whether to load from Hugging Face Hub
        local_path: Local path to the model directory
        hf_token: Hugging Face API token for private repositories
    
    Returns:
        Loaded tokenizer and model
    """
    if use_hf_hub:
        # Use HF Hub path
        if model_type == "original":
            hub_path = f"{DEFAULT_MODEL_REPO}/{DEFAULT_ORIGINAL_MODEL_PATH}"
        else:
            hub_path = f"{DEFAULT_MODEL_REPO}/{DEFAULT_IMPROVED_MODEL_PATH}"
        
        return load_model_from_hf_hub(hub_path, use_auth_token=hf_token)
    else:
        # Use local path
        if local_path:
            return load_model_local(local_path)
        else:
            # Default local paths
            if model_type == "original":
                return load_model_local("bert_model/fine_tuned_roberta")
            else:
                return load_model_local("bert_model/fine_tuned_roberta/fine_tuned_roberta_improved")