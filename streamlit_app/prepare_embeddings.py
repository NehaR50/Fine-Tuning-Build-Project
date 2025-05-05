import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
from tqdm import trange

# Updated project path for your setup
project_root = r'C:\Users\nehar\Downloads\buildG'
streamlit_app_data_path = os.path.join(project_root, 'streamlit_app', 'data')

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Load job postings
job_postings_df = pd.read_parquet(os.path.join(streamlit_app_data_path, 'job_postings.parquet'))
job_titles = job_postings_df['job_posting_title'].to_list()

device = get_device()
print(f'✅ Using device: {device}')

# Load models
default_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device=device)

# 👇 Updated full path to your actual fine-tuned model
fine_tuned_model_path = r'C:\Users\nehar\Downloads\Fine-Tuning-Embedding-Model-Using-Synthetic-Training-Data\fine_tuning\data\trained_models\sentence-transformers-paraphrase-MiniLM-L6-v2_triplet_2025-04-29_12-39-22'
fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)

# Generate embeddings in batches
default_embeddings = []
fine_tuned_embeddings = []

for i in trange(0, len(job_titles), 100):
    chunk = job_titles[i:i+100]
    default_embeddings.append(
        default_model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True, device=device)
    )
    fine_tuned_embeddings.append(
        fine_tuned_model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True, device=device)
    )

# Concatenate and save
default_embeddings = np.concatenate(default_embeddings)
fine_tuned_embeddings = np.concatenate(fine_tuned_embeddings)

print(f"🔹 Default embeddings shape: {default_embeddings.shape}")
print(f"🔹 Fine-tuned embeddings shape: {fine_tuned_embeddings.shape}")

np.save(os.path.join(streamlit_app_data_path, 'default_embeddings.npy'), default_embeddings)
np.save(os.path.join(streamlit_app_data_path, 'fine_tuned_embeddings.npy'), fine_tuned_embeddings)

print("✅ Embeddings saved successfully.")
