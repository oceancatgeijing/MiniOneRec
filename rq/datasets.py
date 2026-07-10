import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    """Original dataset: loads text embeddings from .npy file."""

    def __init__(self, data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)

        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0

        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class EmbDatasetWithCollab(data.Dataset):
    """
    Fusion dataset: loads text embeddings AND collaborative embeddings and
    concatenates them along the feature dimension.

    X_output = Concat(Text_Embedding, Collaborative_Embedding)

    The learnable gate (for controlling fusion ratio) is implemented in the
    RQVAE model, not here — this keeps the dataset as a pure data provider.

    Cold-start items (not in collab vocab) get a zero collaborative vector.
    """

    def __init__(self, data_path, collab_emb_path):
        """
        Args:
            data_path: Path to text embedding .npy file
            collab_emb_path: Path to collaborative embedding .pt file
        """
        self.data_path = data_path
        self.collab_emb_path = collab_emb_path

        # ---- Load text embeddings ----
        self.text_embeddings = np.load(data_path)
        nan_mask = np.isnan(self.text_embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in text embeddings")
            self.text_embeddings[nan_mask] = 0.0
        inf_mask = np.isinf(self.text_embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in text embeddings")
            self.text_embeddings[inf_mask] = 0.0

        self.num_items = self.text_embeddings.shape[0]
        self.text_dim = self.text_embeddings.shape[-1]
        print(f"Loaded text embeddings:  shape={self.text_embeddings.shape}")

        # ---- Load collaborative embeddings ----
        collab_data = torch.load(collab_emb_path, map_location='cpu', weights_only=False)
        self.collab_embeddings_dict = collab_data['embeddings']        # Dict[int, Tensor]
        self.collab_dim = collab_data['vector_size']
        self.cold_start_vector = collab_data['cold_start_vector']       # zero vector
        self.collab_item_count = len(self.collab_embeddings_dict)

        print(f"Loaded collab embeddings: {self.collab_item_count} items, "
              f"dim={self.collab_dim}")

        # ---- Build aligned collab array ----
        self.collab_array = torch.zeros((self.num_items, self.collab_dim), dtype=torch.float32)
        missing_count = 0
        for item_id in range(self.num_items):
            if item_id in self.collab_embeddings_dict:
                self.collab_array[item_id] = self.collab_embeddings_dict[item_id].float()
            else:
                missing_count += 1
        if missing_count > 0:
            print(f"  Cold-start items (zero collab vector): {missing_count}/{self.num_items}")

        # ---- Total input dimension ----
        self.dim = self.text_dim + self.collab_dim
        print(f"Fusion input dim: {self.dim} = text({self.text_dim}) + collab({self.collab_dim})")

    def __getitem__(self, index):
        text_emb = torch.FloatTensor(self.text_embeddings[index])
        collab_emb = self.collab_array[index]  # already zero for cold-start
        fused = torch.cat([text_emb, collab_emb], dim=-1)
        return fused

    def __len__(self):
        return self.num_items
