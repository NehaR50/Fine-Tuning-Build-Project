## Fine-Tuned Job Title Embedding Model using Synthetic Data

This is an interactive Streamlit web application that allows users to search for job titles and discover similar job postings using both a default sentence embedding model and a fine-tuned model trained with synthetic data. This project demonstrates a domain-specific sentence embedding model for comparing job titles more effectively than general-purpose models. It uses synthetic variations of job titles (jittered titles) for fine-tuning, enabling enhanced semantic similarity detection in employment-related applications. A visual interface using Streamlit and clustering visualizations using t-SNE are also provided.


### What This Project Does
  📚 Generates synthetic variations of job titles using LLMs.
  
  🎯 Fine-tunes a lightweight transformer model using triplet loss.
  
  🔍 Builds a search engine that compares job title similarity.
  
  📊 Visualizes how embeddings improve after fine-tuning.
  
  🧠 Benchmarks popular LLM APIs for the same task.


### Live Demo (Optional)
If hosted on Streamlit or Hugging Face Spaces, insert the link here.
Example: 👉 Try it here: https://nehar50-fine-tuning-a-job-title-embeddi-streamlit-appapp-us2ntv.streamlit.app/


### Features


  🔍 Search for job titles and get top matches.

  🤖 Compare results from:
      SentenceTransformer (default)
      Fine-tuned model (trained using triplet loss)

  📊 View similar job titles based on your selected result.

  ⚡ Powered by Sentence Transformers, PyTorch, and Streamlit.
