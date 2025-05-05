import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set your project root
project_root = r'C:\Users\nehar\Downloads\buildG'

# Paths for data and model
test_data_path = os.path.join(project_root, 'fine_tuning', 'data', 'datasets', 'test_ds.csv')
fine_tuned_model_path = os.path.join(project_root, 'fine_tuning', 'data', 'trained_models', r'C:\Users\nehar\Downloads\Fine-Tuning-Embedding-Model-Using-Synthetic-Training-Data\fine_tuning\data\trained_models\sentence-transformers-paraphrase-MiniLM-L6-v2_triplet_2025-04-29_12-39-22')  # Adjust the folder name as needed

# Load test data
test_df = pd.read_csv(test_data_path)

# Device setup
def get_device():
    if torch.cuda.is_available():
        print("CUDA GPU is available. Using GPU...")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Apple MPS is available. Using MPS...")
        return torch.device("mps")
    else:
        print("No GPU found. Using CPU...")
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Load models
try:
    print(f"Loading base model...")
    base_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', device=device)

    print(f"Loading fine-tuned model from {fine_tuned_model_path}...")
    fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=device)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Generate embeddings
jittered_titles = test_df['jittered_title'].to_list()

print("Generating embeddings...")
base_model_embeddings = []
fine_tuned_embeddings = []
batch_size = 16

try:
    for i in tqdm(range(0, len(jittered_titles), 100)):
        chunk = jittered_titles[i:i+100]
        base_model_embeddings.append(base_model.encode(chunk, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, device=device))
        fine_tuned_embeddings.append(fine_tuned_model.encode(chunk, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, device=device))

    base_model_embeddings = np.concatenate(base_model_embeddings)
    fine_tuned_embeddings = np.concatenate(fine_tuned_embeddings)
    print(f"Embeddings shape: {base_model_embeddings.shape}")
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)

# Dimensionality reduction using t-SNE
print("Applying t-SNE dimensionality reduction...")
try:
    tsne = TSNE(n_components=2, random_state=101)
    base_model_embeddings_2d = tsne.fit_transform(base_model_embeddings)
    fine_tuned_embeddings_2d = tsne.fit_transform(fine_tuned_embeddings)
except Exception as e:
    print(f"Error in t-SNE: {e}")
    exit(1)

# Visualization function
def visualize_embeddings(base_embeddings_2d, fine_tuned_embeddings_2d, test_df, subset_size=20):
    random_seed_title_subset = np.random.choice(test_df['seed_title'].unique(), subset_size, replace=False)
    seed_title_mask = test_df['seed_title'].isin(random_seed_title_subset)

    test_df_subset = test_df[seed_title_mask]

    base_model_embeddings_subset_2d = base_embeddings_2d[seed_title_mask, :]
    fine_tuned_embeddings_subset_2d = fine_tuned_embeddings_2d[seed_title_mask, :]

    unique_labels = test_df_subset['seed_title'].unique()

    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    colors = {i: cmap(i) for i in range(len(unique_labels))}
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    max_label_len = 30
    short_labels = {
        label: (label if len(label) <= max_label_len else label[:max_label_len] + '...')
        for label in unique_labels
    }

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.35, 1], wspace=0.2)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[2])
    ax_legend = fig.add_subplot(gs[1])

    for label in unique_labels:
        idx = test_df_subset['seed_title'] == label
        ax_left.scatter(base_model_embeddings_subset_2d[idx, 0], base_model_embeddings_subset_2d[idx, 1],
                      color=color_map[label], s=60)
    ax_left.set_title("Base Model Embedding Space (t-SNE)")
    ax_left.set_xlabel("t-SNE 1")
    ax_left.set_ylabel("t-SNE 2")

    for label in unique_labels:
        idx = test_df_subset['seed_title'] == label
        ax_right.scatter(fine_tuned_embeddings_subset_2d[idx, 0], fine_tuned_embeddings_subset_2d[idx, 1],
                       color=color_map[label], s=60)
    ax_right.set_title("Fine-Tuned Embedding Space (t-SNE)")
    ax_right.set_xlabel("t-SNE 1")
    ax_right.set_ylabel("t-SNE 2")

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                 markerfacecolor=color_map[label], markersize=8,
                 label=short_labels[label])
        for label in unique_labels
    ]
    ax_legend.axis('off')
    legend = ax_legend.legend(handles=handles, loc='center', frameon=False, ncol=1)
    plt.setp(legend.get_texts(), fontsize='small')

    output_path = os.path.join(project_root, 'fine_tuning', 'embedding_visualization.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_path}")

    plt.tight_layout()
    plt.show()

# Run the visualization
print("Creating visualization...")
try:
    visualize_embeddings(base_model_embeddings_2d, fine_tuned_embeddings_2d, test_df)
    print("Done!")
except Exception as e:
    print(f"Error in visualization: {e}")
