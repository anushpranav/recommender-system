import torch
import torch.nn as nn
import torch.nn.functional as F

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model for recommendation.
    Combines matrix factorization with multilayer perceptron.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32]):
        """
        Initialize the NCF model.
        
        Args:
            num_users: Number of unique users in the dataset
            num_items: Number of unique items in the dataset
            embedding_dim: Size of the embedding vectors
            layers: List of layer dimensions for the MLP component
        """
        super(NCFModel, self).__init__()
        
        # User and item embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim  # Concatenated user and item embeddings
        
        for output_size in layers:
            self.fc_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        """
        Forward pass of the NCF model.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted probability of interaction
        """
        # Get user and item embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        # Output prediction
        logits = self.output_layer(x)
        pred = self.sigmoid(logits)
        
        return pred

class RecommendationDataset(torch.utils.data.Dataset):
    """
    Dataset for training the NCF model.
    """
    def __init__(self, interactions_df):
        self.users = torch.tensor(interactions_df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(interactions_df['product_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(interactions_df['interaction'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'label': self.labels[idx]
        }