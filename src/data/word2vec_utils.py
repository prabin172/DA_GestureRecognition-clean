import torch
import numpy as np
from typing import List
import gensim.downloader as api
from gensim.models import KeyedVectors
from pathlib import Path

# --- Configuration Constants ---
WORD2VEC_DIM = 300 
W2V_MODEL_NAME = 'word2vec-google-news-300' 

# 1. Dynamically find the project root (IROS26)
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

# 2. Build the path from the root down
W2V_LOCAL_PATH = PROJECT_ROOT / "trained_models" / "MIL" / f"{W2V_MODEL_NAME}.model"

# Global storage for the Word2Vec model instance once it's loaded
_w2v_model = None

def get_word_vectors(words: List[str]) -> torch.Tensor:
    """
    Loads pre-trained Word2Vec, computes vectors for given labels. 
    Prioritizes loading from W2V_LOCAL_PATH to bypass HPC connectivity issues.
    """
    global _w2v_model

    if _w2v_model is None:
        if W2V_LOCAL_PATH.exists():
            print(f"Loading Word2Vec model from local file: {W2V_LOCAL_PATH.name}")
            try:
                # Load the model directly from the saved file
                _w2v_model = KeyedVectors.load(str(W2V_LOCAL_PATH), mmap='r')
            except Exception as e:
                 print(f"❌ ERROR loading local model file: {e}")
                 _w2v_model = None
        
        if _w2v_model is None:
            # Fallback to online download or random vectors if local file not found/fails
            print("Attempting online download...")
            try:
                _w2v_model = api.load(W2V_MODEL_NAME)
            except Exception:
                print(f"Using Random Vectors. Could not download {W2V_MODEL_NAME}.")
                # Create a minimal fallback model
                _w2v_model = KeyedVectors(vector_size=WORD2VEC_DIM)
                _w2v_model.add_vectors(["fallback"], [np.zeros(WORD2VEC_DIM, dtype=np.float32)])
            
    label_vectors = []
    
    for label in words:
        label_words = label.replace('_', ' ').split()
        word_vectors = []
        
        for word in label_words:
            word = word.lower()
            try:
                vector = _w2v_model[word]
                word_vectors.append(vector)
            except KeyError:
                word_vectors.append(np.zeros(WORD2VEC_DIM, dtype=np.float32))
        
        if word_vectors:
            avg_vector = np.mean(word_vectors, axis=0)
        else:
            avg_vector = np.zeros(WORD2VEC_DIM, dtype=np.float32)
            
        label_vectors.append(avg_vector)

    return torch.from_numpy(np.stack(label_vectors)).float()

if __name__ == '__main__':
    # This block is for testing the download and local saving process.
    print("--- Testing Word2Vec Download and Save Process ---")
    
    # 1. Download the model
    model = api.load(W2V_MODEL_NAME)
    
    # 2. Save the model to the target path structure
    W2V_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(W2V_LOCAL_PATH))
    print(f"\nModel successfully downloaded and saved to: {W2V_LOCAL_PATH}")
    
    # 3. Test loading the local file
    test_load = get_word_vectors(['squat', 'pull'])
    print(f"Local loading successful. Shape: {test_load.shape}")