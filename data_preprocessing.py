import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(users_path, products_path, transactions_path):
    """
    Load the datasets from CSV files.
    """
    users = pd.read_csv(users_path)
    products = pd.read_csv(products_path)
    transactions = pd.read_csv(transactions_path)
    return users, products, transactions

def preprocess_data(users, products, transactions):
    """
    Preprocess the data to create a user-item interaction matrix.
    """
    # Convert IDs to integer indices
    user_mapping = {id: idx for idx, id in enumerate(users['user_id'].unique())}
    product_mapping = {id: idx for idx, id in enumerate(products['product_id'].unique())}
    
    # Create a mapping for reverse lookup (index to original ID)
    reverse_user_mapping = {idx: id for id, idx in user_mapping.items()}
    reverse_product_mapping = {idx: id for id, idx in product_mapping.items()}
    
    # Convert transactions to use indices
    interactions = transactions[['user_id', 'product_id']].copy()
    interactions['user_idx'] = interactions['user_id'].map(user_mapping)
    interactions['product_idx'] = interactions['product_id'].map(product_mapping)
    
    # Create positive samples (purchased products)
    positive_samples = interactions[['user_idx', 'product_idx']].drop_duplicates()
    positive_samples['interaction'] = 1
    
    return (positive_samples, user_mapping, product_mapping, 
            reverse_user_mapping, reverse_product_mapping)

def generate_negative_samples(positive_samples, num_users, num_items, negative_ratio=5):
    """
    Generate negative samples for training.
    For each positive sample, generate 'negative_ratio' negative samples.
    """
    # Create a set of positive interactions for fast lookup
    user_item_set = set([
        (row['user_idx'], row['product_idx']) 
        for idx, row in positive_samples.iterrows()
    ])
    
    negative_samples = []
    
    # For each user
    for user_idx in range(num_users):
        # Get all items this user has interacted with
        interacted_items = [
            item for user, item in user_item_set if user == user_idx
        ]
        
        # Calculate how many negative samples we need for this user
        num_neg_samples = len(interacted_items) * negative_ratio
        
        # Randomly sample non-interacted items
        neg_items = []
        while len(neg_items) < num_neg_samples:
            item = np.random.randint(0, num_items)
            if item not in interacted_items and (user_idx, item) not in user_item_set:
                neg_items.append(item)
                user_item_set.add((user_idx, item))  # Add to prevent duplicates
        
        # Create negative samples
        for item in neg_items:
            negative_samples.append({
                'user_idx': user_idx,
                'product_idx': item,
                'interaction': 0
            })
    
    # Convert to DataFrame
    negative_df = pd.DataFrame(negative_samples)
    
    # Combine positive and negative samples
    full_data = pd.concat([positive_samples, negative_df], ignore_index=True)
    
    return full_data

def split_train_test(full_data, test_size=0.2):
    """
    Split the data into training and testing sets.
    """
    train_data, test_data = train_test_split(
        full_data, test_size=test_size, stratify=full_data['interaction'], random_state=42
    )
    return train_data, test_data

# # Add this code at the end of your file
# users_path = 'data/users.csv'
# products_path = 'data/products.csv'
# transactions_path = 'data/transactions.csv'

# # Load the data using your paths
# users, products, transactions = load_data(users_path, products_path, transactions_path)

# # Process the data
# positive_samples, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping = preprocess_data(users, products, transactions)

# # Generate samples
# num_users = len(user_mapping)
# num_items = len(product_mapping)
# full_data = generate_negative_samples(positive_samples, num_users, num_items)

# # Split into train/test sets
# train_data, test_data = split_train_test(full_data)