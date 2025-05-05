# Fine-Tuned Job Title Embedding Model using Synthetic Data

This is an interactive Streamlit web application that allows users to search for job titles and discover similar job postings using both a default sentence embedding model and a fine-tuned model trained with synthetic data. This project demonstrates a domain-specific sentence embedding model for comparing job titles more effectively than general-purpose models. It uses synthetic variations of job titles (jittered titles) for fine-tuning, enabling enhanced semantic similarity detection in employment-related applications. A visual interface using Streamlit and clustering visualizations using t-SNE are also provided. It combines contrastive learning with LLM-driven data generation, offering a scalable and domain-specific alternative to traditional string matching or off-the-shelf embeddings.

## Project Overview
### Core Components

| Component         | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `synthetic_data/` | LLM-based generation of jittered job title variants                            |
| `fine_tuning/`    | Fine-tuning pipeline using Triplet Loss with `SentenceTransformerTrainer`      |
| `streamlit_app/`  | Interactive search engine that compares base and fine-tuned similarity results |
| `visualization`   | t-SNE plots to evaluate semantic clustering of embeddings                      |


## Data Format

Each row in the training data includes:
seed_title: Canonical job title (e.g., Software Engineer)
jittered_title: A variation (e.g., S/W Developer II (Full Stack))
onet_code: Standardized occupational code for stratified splits


## What This Project Does

  📚 Generates synthetic variations of job titles using LLMs.
  
  🎯 Fine-tunes a lightweight transformer model using triplet loss.
  
  🔍 Builds a search engine that compares job title similarity.
  
  📊 Visualizes how embeddings improve after fine-tuning.
  
  🧠 Benchmarks popular LLM APIs for the same task.


## Live Demo (Optional)
Hosted on Streamlit Cloud. 👉 Try it here: https://nehar50-fine-tuning-a-job-title-embeddi-streamlit-appapp-us2ntv.streamlit.app/


## Features

  🔍 Search for job titles and get top matches.
  
  🤖 Compare results from:
      SentenceTransformer (default)
      Fine-tuned model (trained using triplet loss)
      
  📊 View similar job titles based on your selected result.
  
  ⚡ Powered by Sentence Transformers, PyTorch, and Streamlit.
