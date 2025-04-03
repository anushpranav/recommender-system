import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import argparse

# Import our modules
from data_preprocessing import load_data, preprocess_data, generate_negative_samples, split_train_test
from ncf_model import NCFModel, RecommendationDataset
from training import train_model, evaluate_model, recommend_items

def privacy_accuracy_tradeoff(data, num_users, num_items, epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]):
    """
    Analyze the trade-off between privacy and accuracy.
    
    Args:
        data: Full dataset
        num_users: Number of users
        num_items: Number of items
        epsilons: List of epsilon values to test
    
    Returns:
        DataFrame of results
    """
    # Split data
    train_data, test_data = split_train_test(data, test_size=0.2)
    
    # Create datasets
    train_dataset = RecommendationDataset(train_data)
    test_dataset = RecommendationDataset(test_data)
    
    # Common parameters
    embedding_dim = 64
    layers = [128, 64, 32]
    batch_size = 1024
    epochs = 10
    lr = 0.001
    delta = 1e-5
    
    # Results container
    results = []
    
    # First, train non-private model as baseline
    print("Training baseline model without privacy...")
    model = NCFModel(num_users, num_items, embedding_dim=embedding_dim, layers=layers)
    trained_model, _, test_metrics = train_model(
        model, train_dataset, test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        use_dp=False
    )
    
    # Record baseline results
    results.append({
        'epsilon': float('inf'),  # No privacy
        'privacy_level': 'None',
        'auc': test_metrics[-1]['auc'],
        'precision@10': test_metrics[-1]['precision@10'],
        'recall@10': test_metrics[-1]['recall@10']
    })
    
    # Train with different privacy budgets
    for epsilon in epsilons:
        print(f"Training with privacy budget ε = {epsilon}...")
        
        model = NCFModel(num_users, num_items, embedding_dim=embedding_dim, layers=layers)
        trained_model, _, test_metrics = train_model(
            model, train_dataset, test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            use_dp=True,
            epsilon=epsilon,
            delta=delta
        )
        
        # Determine privacy level label
        if epsilon <= 0.5:
            privacy_level = 'High'
        elif epsilon <= 2.0:
            privacy_level = 'Medium'
        else:
            privacy_level = 'Low'
        
        # Record results
        results.append({
            'epsilon': epsilon,
            'privacy_level': privacy_level,
            'auc': test_metrics[-1]['auc'],
            'precision@10': test_metrics[-1]['precision@10'],
            'recall@10': test_metrics[-1]['recall@10']
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_privacy_accuracy_tradeoff(results):
    """
    Plot the trade-off between privacy and accuracy.
    
    Args:
        results: DataFrame of results from privacy_accuracy_tradeoff()
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Add color coding for privacy levels
    colors = {'High': 'green', 'Medium': 'blue', 'Low': 'orange', 'None': 'red'}
    
    # Plot AUC vs epsilon
    for i, metric in enumerate(['auc', 'precision@10', 'recall@10']):
        sns.lineplot(
            x='epsilon', 
            y=metric, 
            data=results,
            marker='o',
            linewidth=2,
            ax=axes[i]
        )
        
        # Add points with privacy level colors
        for _, row in results.iterrows():
            axes[i].scatter(
                row['epsilon'], 
                row[metric], 
                color=colors[row['privacy_level']],
                s=100,
                edgecolor='black'
            )
        
        # Add labels
        axes[i].set_xscale('log')
        axes[i].set_xlabel('Privacy Budget (ε) - Log Scale')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} vs Privacy Budget')
        
        # Add the point for no privacy (infinity)
        no_privacy = results[results['privacy_level'] == 'None'].iloc[0]
        axes[i].axhline(y=no_privacy[metric], color='red', linestyle='--', alpha=0.5)
        axes[i].text(
            results['epsilon'].min(), 
            no_privacy[metric] * 0.98, 
            f"No Privacy: {no_privacy[metric]:.4f}", 
            color='red'
        )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=level)
        for level, color in colors.items()
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    
    # Set overall title
    fig.suptitle('Privacy-Accuracy Trade-off', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

def category_bias_analysis(trained_model, standard_model, user_mapping, product_mapping, 
                           reverse_product_mapping, products_df, users_df):
    """
    Analyze how differential privacy affects recommendations across different product categories.
    
    Args:
        trained_model: Model trained with differential privacy
        standard_model: Model trained without differential privacy
        user_mapping: Mapping from user ID to index
        product_mapping: Mapping from product ID to index
        reverse_product_mapping: Mapping from index to product ID
        products_df: Products dataframe
        users_df: Users dataframe
    
    Returns:
        DataFrame with analysis results
    """
    print("Analyzing category bias in recommendations...")
    
    # Get all product categories
    categories = products_df['category'].unique()
    
    # Select a sample of users for analysis
    sample_users = np.random.choice(list(user_mapping.keys()), min(50, len(user_mapping)), replace=False)
    
    # Results container
    results = []
    
    # Generate recommendations for each user with both models
    for user_id in sample_users:
        user_idx = user_mapping[user_id]
        
        # Get user info
        user_info = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # Get recommendations from both models
        dp_recommendations = recommend_items(
            trained_model, user_idx, product_mapping, reverse_product_mapping, 
            products_df, num_recommendations=20, exclude_purchased=None
        )
        
        std_recommendations = recommend_items(
            standard_model, user_idx, product_mapping, reverse_product_mapping, 
            products_df, num_recommendations=20, exclude_purchased=None
        )
        
        # Analyze category distribution in recommendations
        dp_category_counts = dp_recommendations['category'].value_counts(normalize=True).to_dict()
        std_category_counts = std_recommendations['category'].value_counts(normalize=True).to_dict()
        
        # Calculate category distribution difference
        for category in categories:
            dp_percent = dp_category_counts.get(category, 0) * 100
            std_percent = std_category_counts.get(category, 0) * 100
            
            results.append({
                'user_id': user_id,
                'gender': user_info['gender'],
                'age': user_info['age'],
                'membership_tier': user_info['membership_tier'],
                'category': category,
                'dp_percent': dp_percent,
                'std_percent': std_percent,
                'difference': dp_percent - std_percent
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_category_bias(results_df):
    """
    Plot the category bias analysis results.
    
    Args:
        results_df: DataFrame from category_bias_analysis()
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Plot average category distribution difference
    category_avg = results_df.groupby('category')[['dp_percent', 'std_percent']].mean().reset_index()
    category_avg = category_avg.melt(
        id_vars=['category'],
        value_vars=['dp_percent', 'std_percent'],
        var_name='model_type',
        value_name='percentage'
    )
    
    # Replace model type names for better readability
    category_avg['model_type'] = category_avg['model_type'].replace({
        'dp_percent': 'With Privacy (DP-SGD)',
        'std_percent': 'Without Privacy'
    })
    
    sns.barplot(
        x='category', 
        y='percentage', 
        hue='model_type',
        data=category_avg,
        ax=axes[0]
    )
    
    axes[0].set_title('Category Distribution in Recommendations')
    axes[0].set_xlabel('Product Category')
    axes[0].set_ylabel('Average Percentage in Top 20')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Plot difference by user demographics
    # Let's focus on age groups
    results_df['age_group'] = pd.cut(
        results_df['age'],
        bins=[0, 25, 35, 45, 100],
        labels=['18-25', '26-35', '36-45', '46+']
    )
    
    demo_diff = results_df.groupby(['age_group', 'category'])['difference'].mean().reset_index()
    
    sns.barplot(
        x='category', 
        y='difference', 
        hue='age_group',
        data=demo_diff,
        ax=axes[1]
    )
    
    axes[1].set_title('Privacy Impact by Age Group')
    axes[1].set_xlabel('Product Category')
    axes[1].set_ylabel('Difference (DP - Standard) %')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set overall title
    fig.suptitle('Impact of Differential Privacy on Recommendation Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def recommendation_diversity_analysis(trained_model, standard_model, user_mapping, product_mapping, 
                                    reverse_product_mapping, products_df):
    """
    Analyze how differential privacy affects the diversity of recommendations.
    
    Args:
        trained_model: Model trained with differential privacy
        standard_model: Model trained without differential privacy
        user_mapping: Mapping from user ID to index
        product_mapping: Mapping from product ID to index
        reverse_product_mapping: Mapping from index to product ID
        products_df: Products dataframe
    
    Returns:
        DataFrame with diversity metrics
    """
    print("Analyzing recommendation diversity...")
    
    # Select a sample of users for analysis
    sample_users = np.random.choice(list(user_mapping.keys()), min(50, len(user_mapping)), replace=False)
    
    # Results container
    results = []
    
    for user_id in sample_users:
        user_idx = user_mapping[user_id]
        
        # Get recommendations from both models
        dp_recommendations = recommend_items(
            trained_model, user_idx, product_mapping, reverse_product_mapping, 
            products_df, num_recommendations=20, exclude_purchased=None
        )
        
        std_recommendations = recommend_items(
            standard_model, user_idx, product_mapping, reverse_product_mapping, 
            products_df, num_recommendations=20, exclude_purchased=None
        )
        
        # Calculate diversity metrics
        # 1. Category diversity (number of unique categories)
        dp_category_diversity = len(dp_recommendations['category'].unique())
        std_category_diversity = len(std_recommendations['category'].unique())
        
        # 2. Brand diversity (number of unique brands)
        dp_brand_diversity = len(dp_recommendations['brand'].unique())
        std_brand_diversity = len(std_recommendations['brand'].unique())
        
        # 3. Price range (max - min)
        dp_price_range = dp_recommendations['price'].max() - dp_recommendations['price'].min()
        std_price_range = std_recommendations['price'].max() - std_recommendations['price'].min()
        
        # 4. Recommendation overlap (Jaccard similarity)
        dp_product_ids = set(dp_recommendations['product_id'])
        std_product_ids = set(std_recommendations['product_id'])
        
        intersection = len(dp_product_ids.intersection(std_product_ids))
        union = len(dp_product_ids.union(std_product_ids))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        results.append({
            'user_id': user_id,
            'dp_category_diversity': dp_category_diversity,
            'std_category_diversity': std_category_diversity,
            'dp_brand_diversity': dp_brand_diversity,
            'std_brand_diversity': std_brand_diversity,
            'dp_price_range': dp_price_range,
            'std_price_range': std_price_range,
            'recommendation_overlap': jaccard_similarity
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_diversity_analysis(diversity_results):
    """
    Plot the diversity analysis results.
    
    Args:
        diversity_results: DataFrame from recommendation_diversity_analysis()
    """
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Category Diversity
    category_data = diversity_results[['dp_category_diversity', 'std_category_diversity']].melt(
        var_name='Model Type',
        value_name='Category Diversity'
    )
    category_data['Model Type'] = category_data['Model Type'].replace({
        'dp_category_diversity': 'With Privacy (DP-SGD)',
        'std_category_diversity': 'Without Privacy'
    })
    
    sns.boxplot(
        x='Model Type', 
        y='Category Diversity',
        data=category_data,
        ax=axes[0, 0]
    )
    
    axes[0, 0].set_title('Category Diversity')
    
    # 2. Brand Diversity
    brand_data = diversity_results[['dp_brand_diversity', 'std_brand_diversity']].melt(
        var_name='Model Type',
        value_name='Brand Diversity'
    )
    brand_data['Model Type'] = brand_data['Model Type'].replace({
        'dp_brand_diversity': 'With Privacy (DP-SGD)',
        'std_brand_diversity': 'Without Privacy'
    })
    
    sns.boxplot(
        x='Model Type', 
        y='Brand Diversity',
        data=brand_data,
        ax=axes[0, 1]
    )
    
    axes[0, 1].set_title('Brand Diversity')
    
    # 3. Price Range
    price_data = diversity_results[['dp_price_range', 'std_price_range']].melt(
        var_name='Model Type',
        value_name='Price Range'
    )
    price_data['Model Type'] = price_data['Model Type'].replace({
        'dp_price_range': 'With Privacy (DP-SGD)',
        'std_price_range': 'Without Privacy'
    })
    
    sns.boxplot(
        x='Model Type', 
        y='Price Range',
        data=price_data,
        ax=axes[1, 0]
    )
    
    axes[1, 0].set_title('Price Range Diversity')
    
    # 4. Recommendation Overlap
    sns.histplot(
        diversity_results['recommendation_overlap'],
        kde=True,
        ax=axes[1, 1]
    )
    
    axes[1, 1].set_title('Recommendation Overlap (Jaccard Similarity)')
    axes[1, 1].set_xlabel('Jaccard Similarity')
    axes[1, 1].set_ylabel('Count')
    
    # Add vertical line at mean
    mean_overlap = diversity_results['recommendation_overlap'].mean()
    axes[1, 1].axvline(x=mean_overlap, color='red', linestyle='--')
    axes[1, 1].text(
        mean_overlap + 0.02, 
        axes[1, 1].get_ylim()[1] * 0.9,
        f'Mean: {mean_overlap:.2f}', 
        color='red'
    )
    
    # Set overall title
    fig.suptitle('Impact of Differential Privacy on Recommendation Diversity', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main(args):
    print("Privacy-Preserving Recommendation Evaluation")
    print("===========================================")
    
    # Load data
    print("Loading data...")
    users, products, transactions = load_data(args.users_path, args.products_path, args.transactions_path)
    
    # Preprocess data
    print("Preprocessing data...")
    positive_samples, user_mapping, product_mapping, reverse_user_mapping, reverse_product_mapping = preprocess_data(users, products, transactions)
    num_users = len(user_mapping)
    num_items = len(product_mapping)
    
    # Generate negative samples
    print("Generating negative samples...")
    full_data = generate_negative_samples(positive_samples, num_users, num_items, negative_ratio=args.negative_ratio)
    
    if args.privacy_accuracy_tradeoff:
        print("Analyzing privacy-accuracy trade-off...")
        results = privacy_accuracy_tradeoff(
            full_data, num_users, num_items, 
            epsilons=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        plot_privacy_accuracy_tradeoff(results)
    
    if args.category_bias or args.recommendation_diversity:
        # We need both models for these analyses
        print("Training standard model (no privacy)...")
        train_data, test_data = split_train_test(full_data)
        train_dataset = RecommendationDataset(train_data)
        test_dataset = RecommendationDataset(test_data)
        
        # Train standard model
        standard_model = NCFModel(num_users, num_items)
        standard_model, _, _ = train_model(
            standard_model, train_dataset, test_dataset,
            use_dp=False, epochs=args.epochs
        )
        
        # Train DP model
        print(f"Training privacy-preserving model (epsilon={args.epsilon})...")
        dp_model = NCFModel(num_users, num_items)
        dp_model, _, _ = train_model(
            dp_model, train_dataset, test_dataset,
            use_dp=True, epsilon=args.epsilon, epochs=args.epochs
        )
        
        if args.category_bias:
            print("Analyzing category bias...")
            bias_results = category_bias_analysis(
                dp_model, standard_model, user_mapping, product_mapping,
                reverse_product_mapping, products, users
            )
            plot_category_bias(bias_results)
        
        if args.recommendation_diversity:
            print("Analyzing recommendation diversity...")
            diversity_results = recommendation_diversity_analysis(
                dp_model, standard_model, user_mapping, product_mapping,
                reverse_product_mapping, products
            )
            plot_diversity_analysis(diversity_results)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Privacy-Preserving Recommendation Evaluation')
    
    # Data paths
    parser.add_argument('--users_path', type=str, default='users.csv', help='Path to users.csv')
    parser.add_argument('--products_path', type=str, default='products.csv', help='Path to products.csv')
    parser.add_argument('--transactions_path', type=str, default='transactions.csv', help='Path to transactions.csv')
    
    # Analysis options
    parser.add_argument('--privacy_accuracy_tradeoff', action='store_true', help='Run privacy-accuracy trade-off analysis')
    parser.add_argument('--category_bias', action='store_true', help='Run category bias analysis')
    parser.add_argument('--recommendation_diversity', action='store_true', help='Run recommendation diversity analysis')
    
    # Parameters
    parser.add_argument('--negative_ratio', type=int, default=5, help='Ratio of negative to positive samples')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Privacy budget for DP model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    
    args = parser.parse_args()
    main(args)