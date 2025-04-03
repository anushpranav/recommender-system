import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import os

# Import our modules
# Assuming the code from previous snippets is saved in separate files
from data_preprocessing import load_data, preprocess_data, generate_negative_samples, split_train_test
from ncf_model import NCFModel, RecommendationDataset
from training import train_model, plot_results, recommend_items

def main(args):
    print("Privacy-Preserving Personalized Product Recommendations")
    print("=====================================================")
    
    # Load data
    print("Loading data...")
    users, products, transactions = load_data(args.users_path, args.products_path, args.transactions_path)
    print(f"Loaded {len(users)} users, {len(products)} products, and {len(transactions)} transactions")
    
    # Preprocess data
    print("Preprocessing data...")
    positive_samples, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping = preprocess_data(users, products, transactions)
    num_users = len(user_mapping)
    num_items = len(product_mapping)
    print(f"Found {len(positive_samples)} positive interactions between {num_users} users and {num_items} items")
    
    # Generate negative samples
    print(f"Generating negative samples with ratio {args.negative_ratio}...")
    full_data = generate_negative_samples(positive_samples, num_users, num_items, negative_ratio=args.negative_ratio)
    print(f"Total dataset size: {len(full_data)} interactions")
    
    # Split data
    print(f"Splitting data with test ratio {args.test_size}...")
    train_data, test_data = split_train_test(full_data, test_size=args.test_size)
    print(f"Training set: {len(train_data)}, Testing set: {len(test_data)}")
    
    # Create datasets
    train_dataset = RecommendationDataset(train_data)
    test_dataset = RecommendationDataset(test_data)
    
    # Initialize model
    print("Initializing NCF model...")
    model = NCFModel(num_users, num_items, embedding_dim=args.embedding_dim, layers=args.layers)
    
    # Train model
    print("Training model...")
    if args.use_dp:
        print(f"Using Differential Privacy with epsilon={args.epsilon}, delta={args.delta}")
    else:
        print("Training without privacy protection")
    
    trained_model, train_losses, test_metrics = train_model(
        model, train_dataset, test_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        use_dp=args.use_dp,
        epsilon=args.epsilon,
        delta=args.delta
    )
    
    # Plot results
    plot_results(train_losses, test_metrics, use_dp=args.use_dp)
    
    # Save the model
    if args.save_model:
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'user_mapping': user_mapping,
            'product_mapping': product_mapping,
            'reverse_user_mapping': reverse_user_mapping,
            'reverse_product_mapping': reverse_product_mapping,
        }, args.model_path)
        print(f"Model saved to {args.model_path}")
    
    # Example recommendation
    if args.show_recommendations:
        # Generate recommendations for a random user
        user_id = np.random.choice(list(user_mapping.keys()))
        user_idx = user_mapping[user_id]
        
        # Get user's purchased items
        user_purchases = positive_samples[positive_samples['user_idx'] == user_idx]['product_idx'].values
        
        print(f"\nGenerating recommendations for user {user_id} (index: {user_idx}):")
        recommendations = recommend_items(
            trained_model, user_idx, product_mapping, reverse_product_mapping, 
            products, num_recommendations=10, exclude_purchased=user_purchases
        )
        
        print("\nTop 10 Recommended Products:")
        print(recommendations[['product_name', 'category', 'brand', 'price', 'score']])
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Privacy-Preserving Product Recommendations')
    
    # Data paths
    parser.add_argument('--users_path', type=str, default='users.csv', help='Path to users.csv')
    parser.add_argument('--products_path', type=str, default='products.csv', help='Path to products.csv')
    parser.add_argument('--transactions_path', type=str, default='transactions.csv', help='Path to transactions.csv')
    
    # Data preprocessing
    parser.add_argument('--negative_ratio', type=int, default=5, help='Ratio of negative to positive samples')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 64, 32], help='Hidden layer dimensions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Privacy parameters
    parser.add_argument('--use_dp', action='store_true', help='Use Differential Privacy')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Privacy budget (epsilon)')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy parameter (delta)')
    
    # Other
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--model_path', type=str, default='recommender_model.pt', help='Path to save model')
    parser.add_argument('--show_recommendations', action='store_true', help='Show example recommendations')
    
    args = parser.parse_args()
    main(args)


'''
# Train with differential privacy (default ε=1.0)
python main.py --users_path data/users.csv --products_path data/products.csv --transactions_path data/transactions.csv --use_dp --save_model --show_recommendations

# Train without differential privacy
python main.py --users_path data/users.csv --products_path data/products.csv --transactions_path data/transactions.csv --save_model


# Run all analyses
python evaluate.py --users_path data/users.csv --products_path data/products.csv --transactions_path data/transactions.csv --privacy_accuracy_tradeoff --category_bias --recommendation_diversity

# Run only privacy-accuracy trade-off analysis
python evaluate.py --privacy_accuracy_tradeoff
'''