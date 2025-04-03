import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def train_model(model, train_data, test_data, batch_size=1024, epochs=20, 
                lr=0.001, use_dp=True, epsilon=1.0, delta=1e-5):
    """
    Train the NCF model with or without differential privacy.
    
    Args:
        model: NCF model
        train_data: Training dataset
        test_data: Testing dataset
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        use_dp: Whether to use differential privacy
        epsilon: Privacy budget (epsilon)
        delta: Privacy parameter (delta)
    
    Returns:
        Trained model, training losses, testing metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Apply differential privacy if requested
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            max_grad_norm=1.0,
            target_epsilon=epsilon,
            target_delta=delta,
        )
        print(f"Using DP-SGD with epsilon={epsilon}, delta={delta}")
    
    # Training loop
    train_losses = []
    test_metrics = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Use BatchMemoryManager for more efficient memory usage with DP
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=128 if use_dp else batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            
            for batch in tqdm(memory_safe_data_loader, desc=f"Epoch {epoch+1}"):
                # Get batch data
                user_indices = batch['user'].to(device)
                item_indices = batch['item'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(user_indices, item_indices).squeeze()
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(memory_safe_data_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        test_metrics.append(evaluate_model(model, test_loader, device))
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - "
              f"Test AUC: {test_metrics[-1]['auc']:.4f} - "
              f"Precision@10: {test_metrics[-1]['precision@10']:.4f} - "
              f"Recall@10: {test_metrics[-1]['recall@10']:.4f}")
        
        # If using DP, print current privacy spent
        if use_dp:
            epsilon = privacy_engine.get_epsilon(delta)
            print(f"Current ε: {epsilon:.2f} (for δ={delta})")
    
    return model, train_losses, test_metrics

def evaluate_model(model, data_loader, device, k=10):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained NCF model
        data_loader: Test data loader
        device: Device to use for evaluation
        k: K value for Precision@K and Recall@K
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            user_indices = batch['user'].to(device)
            item_indices = batch['item'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions
            outputs = model(user_indices, item_indices).squeeze()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate AUC-ROC
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Calculate Precision@K and Recall@K
    # Note: This is a simplified version since we need user-specific calculations
    # In a real system, you'd calculate this per user
    threshold = np.percentile(all_predictions, 100 - (k * 100 / len(all_predictions)))
    binary_preds = (all_predictions >= threshold).astype(int)
    
    precision_k = precision_score(all_labels, binary_preds)
    recall_k = recall_score(all_labels, binary_preds)
    
    return {
        'auc': auc,
        f'precision@{k}': precision_k,
        f'recall@{k}': recall_k
    }

def plot_results(train_losses, test_metrics, use_dp=True):
    """
    Plot training results.
    
    Args:
        train_losses: List of training losses
        test_metrics: List of test metrics
        use_dp: Whether DP was used
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    
    # Plot test metrics
    epochs = range(1, len(test_metrics) + 1)
    ax2.plot(epochs, [m['auc'] for m in test_metrics], label='AUC')
    ax2.plot(epochs, [m['precision@10'] for m in test_metrics], label='Precision@10')
    ax2.plot(epochs, [m['recall@10'] for m in test_metrics], label='Recall@10')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Test Metrics')
    ax2.legend()
    
    # Add title
    title = "Privacy-Preserving NCF" if use_dp else "Standard NCF"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def recommend_items(model, user_idx, product_mapping, reverse_product_mapping, products_df, 
                    num_recommendations=10, exclude_purchased=None):
    """
    Generate product recommendations for a user.
    
    Args:
        model: Trained NCF model
        user_idx: User index to generate recommendations for
        product_mapping: Mapping from product ID to index
        reverse_product_mapping: Mapping from index to product ID
        products_df: Products dataframe
        num_recommendations: Number of recommendations to generate
        exclude_purchased: List of already purchased product indices to exclude
    
    Returns:
        DataFrame of recommended products
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create user tensor (repeated for each item)
    num_items = len(product_mapping)
    user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long).to(device)
    
    # Create item tensor (all items)
    item_indices = list(product_mapping.values())
    item_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor).squeeze().cpu().numpy()
    
    # Create dataframe with predictions
    recommendations_df = pd.DataFrame({
        'product_idx': item_indices,
        'score': predictions
    })
    
    # Convert product_idx back to product_id
    recommendations_df['product_id'] = recommendations_df['product_idx'].map(reverse_product_mapping)
    
    # Exclude already purchased items
    if exclude_purchased is not None:
        recommendations_df = recommendations_df[~recommendations_df['product_idx'].isin(exclude_purchased)]
    
    # Sort by prediction score and take top recommendations
    recommendations_df = recommendations_df.sort_values('score', ascending=False).head(num_recommendations)
    
    # Merge with product details
    result = recommendations_df.merge(products_df, on='product_id')
    
    return result[['product_id', 'product_name', 'category', 'brand', 'price', 'score']]