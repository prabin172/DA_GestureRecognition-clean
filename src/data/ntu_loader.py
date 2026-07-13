import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from pathlib import Path
from src.data.word2vec_utils import get_word_vectors

class UnifiedNTUDataset(Dataset):
    def __init__(self, data_path, mode='supervised', max_frames=120):
        """
        mode: 'supervised' (Returns Label Int), 
              'mil' (Returns Word2Vec Embed), 
              'mae' (Returns just Quats)
        """
        self.data_path = data_path
        self.mode = mode
        self.max_frames = max_frames
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        
        # Load Action ID to Name mapping from JSON
        json_path = Path(__file__).parent / "action_names.json"
        with open(json_path, 'r') as f:
            # We convert keys to int because JSON keys are strings
            self.action_names = {int(k): v for k, v in json.load(f).items()}

        if self.mode == 'mil':
            print("Generating Semantic Label Embeddings...")
            # Use action_id as key to get the text, then get the 300D vector
            unique_ids = sorted(list(set([int(f.split('A')[-1].split('.')[0]) for f in self.file_list])))
            names = [self.action_names.get(i, "action") for i in unique_ids]
            vectors = get_word_vectors(names)
            self.w2v_map = {a_id: vec for a_id, vec in zip(unique_ids, vectors)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_path, file_name)

        # Load data and flatten from (Frames, 17, 4) to (Frames, 68)
        data = np.load(file_path).astype(np.float32)
        num_frames = min(data.shape[0], self.max_frames)
        data_flat = data.reshape(data.shape[0], -1) 
        
        # Temporal Padding and Masking
        padded_x = np.zeros((self.max_frames, 68), dtype=np.float32)
        padded_x[:num_frames, :] = data_flat[:num_frames, :]
        
        mask = np.zeros(self.max_frames, dtype=np.float32)
        mask[:num_frames] = 1.0

        # Above masking is a padding mask, not to be confused with 
        # MAE masking (it will be done in MAE code separately).
        # This mask simply indicates which frames are real vs padded, and is returned for all modes for consistency. 
        # I take this like an upper layer identification.
        # In MAE, the model will further apply its own random masking on top of this temporal padding mask.
        # 1.0 (True): "Hey Transformer, look at this frame, it's a real part of the person moving."
        # 0.0 (False): "Ignore this frame. It’s just a zero-filler I added to make the math work."

        # Extract Action ID
        action_id = int(file_name.split('A')[-1].split('.')[0])
        
        # Return based on Mode
        if self.mode == 'mae':
            return torch.from_numpy(padded_x), torch.from_numpy(mask)
        
        if self.mode == 'mil':
            # Returns Quats, Mask, and the 300D Word2Vec vector
            return torch.from_numpy(padded_x), torch.from_numpy(mask), self.w2v_map[action_id]
            
        # Default: Supervised
        return torch.from_numpy(padded_x), torch.from_numpy(mask), torch.tensor(action_id-1, dtype=torch.long)