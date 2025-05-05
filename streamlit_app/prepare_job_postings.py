import os
import pandas as pd
import numpy as np
import ast

# Updated Project Root for your setup
project_root = r'C:\Users\nehar\Downloads\Fine-Tuning-Embedding-Model-Using-Synthetic-Training-Data'
streamlit_data_path = os.path.join(project_root, 'streamlit_app', 'data')

job_postings_df = pd.read_csv(os.path.join(streamlit_data_path, 'job_postings.csv'))

def extract_first_url(url_str):
    try:
        url_list = ast.literal_eval(url_str)
        return url_list[0] if url_list else None
    except:
        return None

job_postings_df['URL'] = job_postings_df['URL'].apply(extract_first_url)
job_postings_df['POSTED'] = pd.to_datetime(job_postings_df['POSTED'], errors='coerce')

job_postings_df.rename({
    'COMPANY_NAME': 'company',
    'TITLE_RAW': 'job_posting_title',
    'ONET_2019': 'onet_code',
    'ONET_2019_NAME': 'onet_name',
    'POSTED': 'date_posted',
    'URL': 'url'
}, inplace=True, axis=1)

# Sampling can be adjusted based on dataset size
job_postings_df = job_postings_df.sample(frac=(1/6), random_state=42).reset_index(drop=True)

job_postings_df.to_parquet(
    os.path.join(streamlit_data_path, 'job_postings.parquet'),
    engine="pyarrow",
    compression="snappy"
)

print("✅ Job postings saved to Parquet successfully.")
