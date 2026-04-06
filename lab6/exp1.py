import os
import sys
import datetime
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lab6.utils import *
    from lab6.visualization_helper import *
else:
    from .utils import *
    from .visualization_helper import *


def run_custom_comparison(n_users=100, n_influencers=10, n_products=20, seed=42, alpha=10., beta=0.9,
                         score_rule="borda", init_score="custom", convex=True, p=1, metric="agg", verbose=False):
    """
    Run comparison with custom initialization parameters supporting both AggRI and CardRI metrics
    """
    np.random.seed(seed)
    U = np.arange(n_users)
    C = np.arange(n_products)
    
    # Initialize scoring rule
    scoring_rule = borda_rule(n_products)
    if score_rule == "custom":
        scoring_rule = initialize_custom_scoring_rule(n_products)
    elif score_rule == "borda":
        scoring_rule = borda_rule(n_products)
    elif score_rule == "lp":
        scoring_rule = lp_scoring_rule(n_products, convex=convex, p=p)


    W = initialize_custom_network(n_users, n_influencers)
    initial_score = None
    
    # Choose user score initialization based on parameter
    if init_score == "network":
        user_scores, W_, initial_score = initialize_network_optimized_scores(n_users, n_influencers, n_products, 
                                                        scoring_rule=scoring_rule, seed=seed, alpha=alpha, beta=beta)
        # W[:n_users, :n_users] = W_
        if verbose:
            # Calculate statistics to compare user_scores and initial_score
            diff = user_scores - initial_score
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            max_diff = np.max(diff)
            min_diff = np.min(diff)
            mse = np.mean(diff**2)
            
            print(f"Approximation Statistics:")
            print(f"Mean difference: {mean_diff:.4f}")
            print(f"Std deviation: {std_diff:.4f}")
            print(f"Max difference: {max_diff:.4f}")
            print(f"Min difference: {min_diff:.4f}")
            print(f"Mean squared error: {mse:.4f}")
            
            # Visualize the weight matrix W as a heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(W, cmap='viridis', aspect='auto')
            plt.colorbar(label='Weight Value')
            plt.title('Network Weight Matrix (W)', fontsize=20)
            plt.xlabel('User (Target)', fontsize=18)
            plt.ylabel('User (Source)', fontsize=18)
            plt.tight_layout()
            plt.savefig('network_weights_heatmap.png', dpi=300)
            plt.show()
            
    elif init_score == "custom":
        user_scores = initialize_custom_user_scores(n_users, n_influencers, n_products, 
                                                  scoring_rule=scoring_rule, seed=seed, 
                                                  alpha=alpha, beta=beta)
    elif init_score == "twotype":
        user_scores = initialize_two_type_user_scores(n_users, n_influencers, n_products, scoring_rule=scoring_rule,  alpha=alpha, beta=beta, type_ratios=[0.6, 0.4], seed=seed)
    elif init_score == "multitype":
        user_scores = initialize_multi_type_user_scores(n_users, n_influencers, n_products, scoring_rule=scoring_rule, alpha=alpha, beta=beta, n_types=4, type_ratios=[0.25, 0.25, 0.25, 0.25], seed=seed)
    elif init_score == "maxtype":
        user_scores = initialize_multi_type_user_scores(n_users, n_influencers, n_products, scoring_rule=scoring_rule, alpha=alpha, beta=beta, n_types=10, type_ratios=[0.1]*10, seed=seed)    

    # Use specified metric for max score calculation
    max_score = compute_max_score(U, W, C, n_influencers, user_scores, metric)
    
    # Select appropriate algorithm based on metric
    strategies = {
        'Random': random_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
        'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores, metric),
        'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores, metric),
        # Use CAGGRIM for 'agg' metric and CARDRIM for 'card' metric
        'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores) if metric == 'agg' else cardrim(U, W, C, n_influencers, scoring_rule, user_scores)
    }
    
    results = {}
    for name, rankings in strategies.items():
        agg_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, 'agg')
        card_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, 'card')
        results[name] = {
            'AggRI': agg_imp,
            'CardRI': card_imp,
            'Rankings': rankings,
            # Use the appropriate metric for ratio calculation
            'Ratio': (agg_imp if metric == 'agg' else card_imp) / max_score if max_score > 0 else 0
        }
    
    return results, user_scores, W, scoring_rule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run influence maximization experiments.')
    parser.add_argument('--n_users', type=int, default=100, help='Number of users')
    parser.add_argument('--n_influencers', type=int, default=10, help='Number of influencers')
    parser.add_argument('--n_products', type=int, default=22, help='Number of products')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init_score', type=str, default="custom", choices=["network", "custom", "twotype", "multitype", "maxtype"], help='User score initialization method')
    parser.add_argument('--score_rule', type=str, default="borda", choices=["borda", "lp", "custom"], help='Scoring rule to use')
    parser.add_argument('--convex', type=lambda x: (str(x).lower() == 'true'), default=True, help=' Convexity for lp scoring rule (True/False)') # Handle boolean string
    parser.add_argument('--p', type=float, default=2.0, help='P-value for lp scoring rule')
    parser.add_argument('--metric', type=str, default="agg", choices=["agg", "card"], help='Metric to use (agg/card)')
    parser.add_argument('--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable verbose output (True/False)') # Handle boolean string
    
    args = parser.parse_args()

    # Parameters
    # n_users = 100 # Old parameter initialization
    # n_influencers = 10 # Old parameter initialization
    # n_products = 22 # Old parameter initialization
    # seed = 42 # Old parameter initialization
    
    # Define experiment parameters using argparse arguments
    params = {
        'n_users': args.n_users,
        'n_influencers': args.n_influencers,
        'n_products': args.n_products,
        'seed': args.seed,
        'init_score': args.init_score,
        'score_rule': args.score_rule, 
        'convex': args.convex,
        'p': args.p,
        'metric': args.metric,
        'verbose': args.verbose
        # 'prefix': "fix"
    }
    
    # Create a more descriptive directory name
    score_type = f"{params['score_rule']}"
    if params['score_rule'] == "lp":
        curve_type = "convex" if params['convex'] else "concave"
        score_type = f"{score_type}_{curve_type}{params['p']}"
    
    init_type = f"{params['init_score']}"
    metric = params['metric']
    prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    results_dir = f'./lab6/results/lab6exp1-{prefix}__{metric}__{score_type}__{init_type}-init'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save parameters to a JSON file
    param_file_path = os.path.join(results_dir, "parameters.json")
    with open(param_file_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to: {param_file_path}")

    # Alpha and Beta values to test
    # alpha_values = np.linspace(0.55, 0.99, 10)
    alpha_values = np.arange(0.6, 1.05, 0.1)
    print(f"Testing alpha-beta values as follows:")

    # Generate beta values with extended range for each alpha
    all_beta_values = {}
    for alpha in alpha_values:
        beta_max = alpha * params['n_influencers'] - 0.015
        # low_beta_values = np.linspace(0.11, 0.99, 9)
        low_beta_values = np.arange(0.2, 1.0, 0.1)
        # Create high beta values with .5 increments (1.0, 1.5, 2.0, 2.5, etc.)
        high_beta_values = np.arange(1.0, beta_max + 0.5, 1.0)
        all_beta_values[alpha] = np.unique(np.concatenate((low_beta_values, high_beta_values)))
        
        # high_beta_values = np.linspace(4.5, beta_max, 20)
        # all_beta_values[alpha] = high_beta_values
        

        print(f"Alpha: {alpha:.2f}, Beta range: 0.11 to {beta_max:.2f}, Total points: {len(all_beta_values[alpha])}")

    # Strategies to compare
    strategies = ['Random', 'Independent', 'Sequential', 'CAGGRIM']

    # print(f"Testing alpha values: {alpha_values}")
    print(f"Using metric: {metric}")
    print(f"Score rule: {score_type}")
    print(f"Init type: {init_type}")
    print(f"Results will be saved to: {results_dir}")

    # Create a dictionary to store all results
    all_results = {}
    all_ratios = {}

    # Process all alpha and beta combinations
    for alpha_idx, alpha in enumerate(alpha_values):
        print(f"\n--- Testing with alpha = {alpha:.2f} ({alpha_idx+1}/{len(alpha_values)}) ---")
        
        all_results[alpha] = {strategy: [] for strategy in strategies}
        all_ratios[alpha] = {strategy: [] for strategy in strategies}
        beta_values = all_beta_values[alpha]
        
        for beta_idx, beta in enumerate(beta_values):
            # Update alpha and beta in params
            params['alpha'] = float(alpha)
            params['beta'] = float(beta)
            
            # Run comparison with specified parameters
            results, user_scores, W, scoring_rule = run_custom_comparison(**params)
            
            # Extract and store results based on the metric
            for strategy in strategies:
                if strategy in results:
                    if metric == 'agg':
                        all_results[alpha][strategy].append(results[strategy]['AggRI'])
                    else:  # metric == 'card'
                        all_results[alpha][strategy].append(results[strategy]['CardRI'])
                    all_ratios[alpha][strategy].append(results[strategy]['Ratio'])

        # Create plots for this alpha value with the specified metric
        plt.figure(figsize=(12, 8))

        # Plot scores
        plt.subplot(2, 1, 1)
        for strategy in strategies:
            if len(all_results[alpha][strategy]) > 0:
                plt.plot(beta_values, all_results[alpha][strategy], marker='o', label=strategy)

        plt.title(f'Strategy Performance vs Beta (Alpha={alpha:.2f}, {metric.upper()} Metric)', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel(f'{metric.upper()}RI Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)

        # Plot ratio to max scores
        plt.subplot(2, 1, 2)
        for strategy in strategies:
            if len(all_ratios[alpha][strategy]) > 0:
                plt.plot(beta_values, all_ratios[alpha][strategy], marker='o', label=strategy)

        plt.title(f'Ratio to Max Score vs Beta (Alpha={alpha:.2f}, {metric.upper()} Metric)', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel(f'Ratio to Max Score ({metric.upper()})', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/beta_sensitivity_{metric}_alpha_{alpha:.2f}.png', dpi=300)
        plt.close()
        print(f"  Saved {metric} strategy performance plot for alpha={alpha:.2f}")

        # Create relative performance plot
        plt.figure(figsize=(12, 6))
        caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
        if len(caggrim_scores) > 0:
            for strategy in strategies:
                if strategy != 'CAGGRIM' and len(all_results[alpha][strategy]) > 0:
                    relative_scores = np.array(all_results[alpha][strategy]) / caggrim_scores
                    plt.plot(beta_values, relative_scores, marker='o', label=f"{strategy} / CAGGRIM")

            plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
            plt.title(f'Strategy Performance Relative to CAGGRIM (Alpha={alpha:.2f}, {metric.upper()})', fontsize=20)
            plt.xlabel('Beta value', fontsize=18)
            plt.ylabel('Relative Performance', fontsize=18)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/beta_sensitivity_relative_{metric}_alpha_{alpha:.2f}.png', dpi=300)
            plt.close()
            print(f"  Saved relative {metric} performance plot for alpha={alpha:.2f}")

    print(f"\nCompleted all alpha and beta combinations for {metric.upper()} metric.")

    # Generate summary visualizations
    if len(all_results) == len(alpha_values):
        # Generate basic summary visualizations
        success = generate_summary_visualizations(all_results, all_ratios, alpha_values, all_beta_values, strategies, results_dir)
        
        # Generate relative performance visualizations
        rel_success = create_relative_performance_visualizations(all_results, all_ratios, alpha_values, all_beta_values, strategies, results_dir)
        
        if success and rel_success:
            print(f"\nAll {metric.upper()} visualizations generated successfully!")
        else:
            print("\nThere were issues generating some visualizations.")
    else:
        print("\nSkipping visualizations as not all alpha values were processed successfully.")

    print(f"All results have been saved to: {results_dir}")
