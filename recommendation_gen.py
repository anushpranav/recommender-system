import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ncf_model import NCFModel  # Make sure to import your model class

def load_model_and_mappings(model_path):
    """Load the trained model and mappings."""
    checkpoint = torch.load(model_path)
    
    # Get mappings
    user_mapping = checkpoint['user_mapping']
    product_mapping = checkpoint['product_mapping']
    reverse_user_mapping = checkpoint['reverse_user_mapping']
    reverse_product_mapping = checkpoint['reverse_product_mapping']
    
    # Initialize model
    num_users = len(user_mapping)
    num_items = len(product_mapping)
    model = NCFModel(num_users, num_items)  # Use the same architecture as during training
    
    # Fix the state dict by removing the '_module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_module.'):
            new_key = key[8:]  # Remove '_module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load model weights
    model.load_state_dict(new_state_dict)
    model.eval()  # Set to evaluation mode
    
    return model, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping

def generate_recommendations_for_all_users(model, user_mapping, reverse_product_mapping, 
                                          products_df, top_n=10, batch_size=128):
    """Generate recommendations for all users."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Prepare to store all recommendations
    all_recommendations = []
    
    # Get all product indices
    num_items = len(product_mapping)
    all_item_indices = list(product_mapping.values())
    all_item_tensor = torch.tensor(all_item_indices, dtype=torch.long)
    
    # Process users in batches to avoid memory issues
    user_indices = list(user_mapping.values())
    
    for i in tqdm(range(0, len(user_indices), batch_size), desc="Generating recommendations"):
        batch_users = user_indices[i:i+batch_size]
        
        # Create tensors for batch processing
        batch_recommendations = []
        
        for user_idx in batch_users:
            # Create user tensor (repeated for each item)
            user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long).to(device)
            item_tensor = all_item_tensor.to(device)
            
            # Get predictions
            with torch.no_grad():
                predictions = model(user_tensor, item_tensor).squeeze().cpu().numpy()
            
            # Get top N recommendations
            top_indices = np.argsort(predictions)[-top_n:][::-1]
            top_item_indices = [all_item_indices[i] for i in top_indices]
            top_scores = predictions[top_indices]
            
            # Convert indices back to IDs
            user_id = next(k for k, v in user_mapping.items() if v == user_idx)
            product_ids = [reverse_product_mapping[idx] for idx in top_item_indices]
            
            # Create recommendations for this user
            for i, (product_id, score) in enumerate(zip(product_ids, top_scores)):
                batch_recommendations.append({
                    'user_id': user_id,
                    'rank': i+1,
                    'product_id': product_id,
                    'score': score
                })
        
        # Add batch recommendations to the full list
        all_recommendations.extend(batch_recommendations)
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(all_recommendations)
    
    # Optionally merge with product details
    if products_df is not None:
        recommendations_df = recommendations_df.merge(
            products_df[['product_id', 'product_name', 'category', 'brand', 'price']], 
            on='product_id'
        )
    
    return recommendations_df

if __name__ == "__main__":
    # Load data
    products_path = 'data/products.csv'
    products = pd.read_csv(products_path)
    
    # Load model and mappings
    model_path = 'recommender_model.pt'
    model, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping = load_model_and_mappings(model_path)
    
    # Generate recommendations for all users
    all_recommendations = generate_recommendations_for_all_users(
        model, user_mapping, reverse_product_mapping, products, top_n=10
    )
    
    # Save recommendations to CSV
    all_recommendations.to_csv('all_user_recommendations.csv', index=False)
    print(f"Generated recommendations for {len(user_mapping)} users")
    print(f"Total recommendations: {len(all_recommendations)}")
    
    # Display sample recommendations
    print("\nSample recommendations:")
    sample_user = list(user_mapping.keys())[0]
    print(f"For user {sample_user}:")
    print(all_recommendations[all_recommendations['user_id'] == sample_user])