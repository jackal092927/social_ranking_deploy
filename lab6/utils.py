import numpy as np
from scipy.interpolate import interp1d
from curve_function import lp_curve
import matplotlib.pyplot as plt

def calc_matrix(U, W, user_scores):
    """Calculates residual matrix including target column"""
    n_users = len(U)
    scores = W[:n_users, :] @ user_scores  # Shape: (n_users, n_products)
    target_scores = scores[:, 0:1]  # Keep dimension using 0:1
    M = np.maximum(0, scores - target_scores)
    return M

def greedy_agg(U, W, M, remaining_mask, pos, k, scoring_rule, n_users, tiebreak="random"):
    score_diff = scoring_rule[0] - scoring_rule[pos-1]
    total_influence = score_diff * W[n_users:n_users+k, :].sum(axis=0)
    total_influence = total_influence.reshape(-1, 1)

    # gains = np.sum(total_influence > M[:, remaining_mask], axis=0)

    ## Only consider cases where M is non-zero and total influence exceeds M
    M_subset = M[:, remaining_mask]
    nonzero_mask = M_subset > 0
    comparison = total_influence >= M_subset
    gains = np.sum(comparison & nonzero_mask, axis=0)
    # flag = True
    ## If no gains
    if np.sum(gains) == 0:
        # if flag:
        #     print("No gains at pos", pos)
        #     flag = False
        if tiebreak == "random":
            diff = M_subset - total_influence
            diff[M_subset<=0]=0
            diff_sum = np.sum(diff, axis=0)
            nonzero_idx = np.where(diff_sum!=0)[0]
            if len(nonzero_idx) > 0:
                return np.random.choice(np.where(diff_sum!=0)[0])
            else:
                return np.random.choice(np.where(gains == gains.max())[0])
                # return np.random.choice(np.where(M_subset==0)[0])
        elif tiebreak == "mingap":
            diff = M_subset-total_influence
            diff_sum = np.min(diff, axis=0)
            diff_sum[diff_sum==0] = np.inf
            return np.random.choice(np.where(diff_sum == diff_sum.min())[0])
        
        # diff_sum[diff_sum==0]=-np.inf
        # return np.argmax(diff_sum)

    # return np.argmax(gains)
    ## break the tie randomly

    return np.random.choice(np.where(gains == gains.max())[0])
# np.random.choice(np.where(b == b.max())[0])


def greedy_card(U, W, M, remaining_item_mask, remaining_user_mask, pos, k, scoring_rule, n_users, tiebreak="random"):
    score_diff = scoring_rule[0] - scoring_rule[pos-1]
    total_influence = score_diff * W[n_users:n_users+k, :].sum(axis=0)
    total_influence = total_influence.reshape(-1, 1)
    
    # Only consider users who haven't been covered yet
    M_subset = M[remaining_user_mask, :][:, remaining_item_mask]
    
    # Filter total_influence to only include remaining users
    total_influence_subset = total_influence[remaining_user_mask]
    
    # Only consider cases where M is non-zero and total influence exceeds M
    nonzero_mask = M_subset > 0
    coverage_subset = (total_influence_subset >= M_subset) & nonzero_mask
    gains = np.sum(coverage_subset, axis=0)
    
    # If no gains, use tiebreaking
    if np.max(gains) == 0:
        if tiebreak == "random":
            diff = M_subset - total_influence_subset
            diff[M_subset <= 0] = 0
            diff_sum = np.sum(diff, axis=0)
            nonzero_idx = np.where(diff_sum != 0)[0]
            if len(nonzero_idx) > 0:
                chosen_idx = np.random.choice(nonzero_idx)
            else:
                chosen_idx = np.random.choice(range(len(gains)))
        elif tiebreak == "mingap":
            diff = M_subset - total_influence_subset
            diff_sum = np.min(diff, axis=0)
            diff_sum[diff_sum == 0] = np.inf
            chosen_idx = np.random.choice(np.where(diff_sum == diff_sum.min())[0])
    else:
        # Use random choice among best options to break ties
        chosen_idx = np.random.choice(np.where(gains == gains.max())[0])
    
    # Convert the subset coverage to full user coverage mask
    # Initialize a mask for all users
    full_coverage = np.zeros(len(U), dtype=bool)
    
    # Get indices of remaining users
    remaining_user_indices = np.where(remaining_user_mask)[0]
    
    # Set the covered users in the full mask
    if len(remaining_user_indices) > 0:
        full_coverage[remaining_user_indices] = coverage_subset[:, chosen_idx]
    
    return chosen_idx, full_coverage

def caggrim(U, W, C, k, scoring_rule, user_scores):
    n_users = len(U)
    n_products = len(C)
    M = calc_matrix(U, W, user_scores)
    
    rankings = np.zeros((k, n_products), dtype=int)
    remaining_mask = np.ones(n_products, dtype=bool)
    remaining_mask[0] = False  # Target product is index 0
    
    for pos in range(n_products-1, 0, -1):
        best_idx = greedy_agg(U, W, M, remaining_mask, pos+1, k, scoring_rule, n_users)
        abs_idx = np.where(remaining_mask)[0][best_idx]
        rankings[:, pos] = abs_idx
        remaining_mask[abs_idx] = False
    
    return rankings

def cardrim(U, W, C, k, scoring_rule, user_scores):
    n_users = len(U)
    n_products = len(C)
    M = calc_matrix(U, W, user_scores)
    
    rankings = np.zeros((k, n_products), dtype=int)
    remaining_item_mask = np.ones(n_products, dtype=bool)
    remaining_item_mask[0] = False  # Target product is index 0
    remaining_user_mask = np.ones(n_users, dtype=bool)  # Track users who haven't been covered
    
    for pos in range(n_products-1, 0, -1):
        best_idx, covered_user_mask = greedy_card(U, W, M, remaining_item_mask, remaining_user_mask, pos+1, 
                                                 k, scoring_rule, n_users, tiebreak="random")
        abs_idx = np.where(remaining_item_mask)[0][best_idx]
        rankings[:, pos] = abs_idx
        remaining_item_mask[abs_idx] = False
        
        # Update which users have been covered - directly using the mask
        remaining_user_mask = remaining_user_mask & ~covered_user_mask
        
        # If all users are covered, we can stop
        if not np.any(remaining_user_mask):
            # Fill remaining positions randomly
            for p in range(pos-1, 0, -1):
                avail_idx = np.random.choice(np.where(remaining_item_mask)[0])
                rankings[:, p] = avail_idx
                remaining_item_mask[avail_idx] = False
            break
        
    return rankings

def independent_strategy(U, W, C, k, scoring_rule, user_scores, metric='agg'):
    """
    Independent strategy that can use either agg or card metric
    """
    n_users = len(U)
    n_products = len(C)
    rankings = np.zeros((k, n_products), dtype=int)
    
    for i in range(k):
        # Create single influencer weight matrix
        single_W = np.zeros((n_users + 1, n_users))
        single_W[:n_users, :] = W[:n_users, :]
        single_W[-1, :] = W[n_users + i, :]
        
        # Run appropriate algorithm based on metric
        if metric == 'agg' or metric == 'cagg':
            single_ranking = caggrim(U, single_W, C, 1, scoring_rule, user_scores)
        else:  # metric == 'card'
            single_ranking = cardrim(U, single_W, C, 1, scoring_rule, user_scores)
        
        rankings[i] = single_ranking[0]
    
    return rankings

def sequential_strategy(U, W, C, k, scoring_rule, user_scores, metric='agg'):
    """
    Sequential strategy that can use either agg or card metric
    """
    n_users = len(U)
    n_products = len(C)
    rankings = np.zeros((k, n_products), dtype=int)
    current_scores = user_scores.astype(float)
    
    for i in range(k):
        single_W = np.zeros((n_users + 1, n_users))
        single_W[:n_users, :] = W[:n_users, :]
        single_W[-1, :] = W[n_users + i, :]
        
        # Run appropriate algorithm based on metric
        if metric == 'agg' or metric == 'cagg':
            single_ranking = caggrim(U, single_W, C, 1, scoring_rule, current_scores)
        else:  # metric == 'card'
            single_ranking = cardrim(U, single_W, C, 1, scoring_rule, current_scores)
            
        rankings[i] = single_ranking[0]
        
        # Fix broadcasting: influence shape (n_users,1)
        influence = W[n_users + i, :].reshape(-1, 1)
        for pos, prod in enumerate(rankings[i]):
            current_scores[:, prod] = current_scores[:, prod] + scoring_rule[pos] * influence.flatten()
    
    return rankings

def random_strategy(U, W, C, k, scoring_rule, user_scores):
    n_users = len(U)
    n_products = len(C)
    rankings = np.zeros((k, n_products), dtype=int)

    ## product 0 ranked first by default
    rankings[:, 0] = 0

    ## rank the rest of the products randomly
    for i in range(k):
        rankings[i, 1:] = np.random.permutation(C[1:])

    return rankings



def evaluate_rankings(U, W, rankings, scoring_rule, user_scores, metric='agg'):
    """
    Evaluate rankings using specified metric (agg or card)
    """
    n_users = len(U)
    k = rankings.shape[0]
    
    # Calculate original scores and rankings
    orig_scores = W[:n_users, :] @ user_scores
    new_scores = orig_scores.copy()
    
    # Apply influencer effects
    # for i in range(k):
    #     influence = W[n_users+i, :].reshape(-1, 1)
    #     for pos, prod in enumerate(rankings[i]):
    #         new_scores[:, prod] = new_scores[:, prod] + scoring_rule[pos] * influence.flatten()



    # Create influencer score matrix (k x n_products)
    # Apply all influencer effects at once
    influencer_scores = np.zeros((k, user_scores.shape[1]))
    for i in range(k):
        influencer_scores[i, rankings[i]] = scoring_rule
    influence_weights = W[n_users:n_users+k, :]  # (k x n_users)
    total_influence = influence_weights.T @ influencer_scores  # (n_users x n_products)
    new_scores += total_influence
    
    # Calculate rankings before and after influence
    orig_ranks = np.argsort(np.argsort(-orig_scores, axis=1), axis=1)
    new_ranks = np.argsort(np.argsort(-new_scores, axis=1), axis=1)
    
    # Calculate rank improvements
    target_rank_improvements = orig_ranks[:, 0] - new_ranks[:, 0]
    
    if metric == 'agg':
        return np.sum(target_rank_improvements)
    else:  # metric == 'card'
        return np.sum(target_rank_improvements > 0)

def run_comparison(n_users=10, n_influencers=5, n_products=4, seed=42):
    np.random.seed(seed)
    U = np.arange(n_users)
    C = np.arange(n_products)
    scoring_rule = np.arange(n_products, 0, -1)
    
    # Initialize user scores as random permutations of scoring rule
    user_scores = np.zeros((n_users, n_products))
    for i in range(n_users):
        user_scores[i] = np.random.permutation(scoring_rule)
        
    W = np.random.random((n_users + n_influencers, n_users))
    
    strategies = {
        'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'CARDRIM': cardrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
        'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores)
    }
    
    results = {}
    for name, rankings in strategies.items():
        agg_imp, card_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores)
        results[name] = {'AggRI': agg_imp, 'CardRI': card_imp, 'Rankings': rankings}
    
    return results

def initialize_custom_user_scores(n_users, n_influencers, n_products, scoring_rule=None, alpha=0.99, beta=0.9, seed=None):
    """
    Initialize user scores according to specified pattern:
    - c1 (index 1) scored 100
    - c0 (index 0) scored 20
    - One random item scored 60
    - Rest scored with decreasing small values
    """
    if seed is not None:
        np.random.seed(seed)
        
    user_scores = np.zeros((n_users, n_products))
    max_gap = scoring_rule[0] - scoring_rule[-1]
    
    for i in range(n_users):
        
        # print("max_gap: ", max_gap)
        # Set fixed scores for c1 and c0
        # user_scores[i, 1] = 519.9  # c1 gets highest score
        # user_scores[i, 0] = 20   # c0 gets 20
        user_scores[i, 0] = 20
        user_scores[i,1] = alpha * n_influencers * max_gap  + user_scores[i, 0] - 0.1 # 20+19*9-0.1
        # print(user_scores[i,1]) 

        
        # Pick random item (excluding c0 and c1) for score 60
        available_indices = list(range(2, n_products))
        second_choice = np.random.choice(available_indices)
        # user_scores[i, second_choice] = 60
        user_scores[i, second_choice] = user_scores[i, 0] + beta * max_gap
        # user_scores[i, second_choice] = user_scores[i, 0] + beta * 2
        
        # Remove chosen index from available indices
        available_indices.remove(second_choice)
        
        # Randomly assign remaining positions with small increasing scores
        np.random.shuffle(available_indices)
        for idx, pos in enumerate(available_indices):
            user_scores[i, pos] = idx * 0.01
            
    return user_scores

def initialize_two_type_user_scores(n_users, n_influencers, n_products, scoring_rule=None, 
                                   alpha=0.99, beta=0.9, type_ratios=[0.4, 0.6], seed=None):
    """
    Initialize user scores with two distinct user types:
    
    Type 1 users (type_ratios[0] portion of all users):
    - Product 0 (target product) always ranked third
    - Product 1 ranked first (highest score)
    - One random product from products 2-9 ranked second
    - All other products ranked randomly
    
    Type 2 users (type_ratios[1] portion of all users):
    - Product 0 (target product) always ranked third
    - Product 19 ranked first (highest score)
    - One random product from products 10-18 ranked second
    - All other products ranked randomly
    
    Args:
        n_users: Number of users
        n_influencers: Number of influencers
        n_products: Number of products
        scoring_rule: Scoring rule array
        alpha: Factor for highest score calculation
        beta: Factor for second-highest score calculation
        type_ratios: Proportion of type 1 and type 2 users [type1_ratio, type2_ratio]
        seed: Random seed
        
    Returns:
        Array of user scores (n_users, n_products)
    """
    if seed is not None:
        np.random.seed(seed)
        
    assert len(type_ratios) == 2 and abs(sum(type_ratios) - 1.0) < 1e-6, "Type ratios must be two values that sum to 1"
    
    user_scores = np.zeros((n_users, n_products))
    max_gap = scoring_rule[0] - scoring_rule[-1]
    
    # Calculate number of users of each type
    n_type1 = int(n_users * type_ratios[0])
    n_type2 = n_users - n_type1
    
    # Generate scores for Type 1 users
    for i in range(n_type1):
        # Product 0 (target) gets third place score
        user_scores[i, 0] = 20
        
        # Product 1 gets highest score
        user_scores[i, 1] = alpha * n_influencers * max_gap + user_scores[i, 0] - 0.1
        
        # Randomly pick one product from 2-9 for second place
        available_indices = list(range(2, min(10, n_products)))
        if available_indices:  # Check if the range is valid
            second_choice = np.random.choice(available_indices)
            user_scores[i, second_choice] = user_scores[i, 0] + beta * max_gap
            
            # Remove chosen index from available indices for random ranking
            remaining_indices = list(range(2, n_products))
            remaining_indices.remove(second_choice)
        else:
            # If products 2-9 don't exist, use what's available
            remaining_indices = list(range(2, n_products))
        
        # Randomly assign scores to remaining positions
        np.random.shuffle(remaining_indices)
        for idx, pos in enumerate(remaining_indices):
            user_scores[i, pos] = idx * 0.01
    
    # Generate scores for Type 2 users
    for i in range(n_type1, n_users):
        # Product 0 (target) gets third place score
        user_scores[i, 0] = 20
        
        # Product 19 (or last product if fewer) gets highest score
        last_idx = min(19, n_products - 1)
        user_scores[i, last_idx] = alpha * n_influencers * max_gap + user_scores[i, 0] - 0.1
        
        # Randomly pick one product from 10-18 for second place
        min_idx = min(10, n_products)
        max_idx = min(19, n_products)
        
        available_indices = list(range(min_idx, max_idx))
        if available_indices:  # Check if the range is valid
            second_choice = np.random.choice(available_indices)
            user_scores[i, second_choice] = user_scores[i, 0] + beta * max_gap
            
            # Remove chosen index and last_idx from available indices for random ranking
            remaining_indices = list(range(1, n_products))
            if second_choice in remaining_indices:
                remaining_indices.remove(second_choice)
            if last_idx in remaining_indices:
                remaining_indices.remove(last_idx)
        else:
            # If products 10-18 don't exist, use what's available
            remaining_indices = list(range(1, n_products))
            if last_idx in remaining_indices:
                remaining_indices.remove(last_idx)
        
        # Randomly assign scores to remaining positions
        np.random.shuffle(remaining_indices)
        for idx, pos in enumerate(remaining_indices):
            user_scores[i, pos] = idx * 0.01
            
    return user_scores


def initialize_multi_type_user_scores(n_users, n_influencers, n_products, n_types=2, 
                                     scoring_rule=None, alpha=0.99, beta=0.9, 
                                     type_ratios=None, seed=None):
    """
    Initialize user scores with multiple distinct user types, with a simplified approach:
    - Each type has a unique first-ranked product (from 1 to n_types)
    - Each type has a distinct set of products to choose from for second place
    - Product 0 (target) is always ranked third for all users
    
    Args:
        n_users: Number of users
        n_influencers: Number of influencers
        n_products: Number of products
        n_types: Number of different user types to create
        scoring_rule: Scoring rule array
        alpha: Factor for highest score calculation
        beta: Factor for second-highest score calculation
        type_ratios: List of ratios for each type (must sum to 1.0)
        seed: Random seed
        
    Returns:
        Array of user scores (n_users, n_products)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default to equal ratios if not specified
    if type_ratios is None:
        type_ratios = [1/n_types] * n_types
    
    assert len(type_ratios) == n_types, "Number of ratios must match number of types"
    assert abs(sum(type_ratios) - 1.0) < 1e-6, "Type ratios must sum to 1"
    assert n_products >= n_types + 1, "Need at least n_types+1 products (plus product 0)"
    
    user_scores = np.zeros((n_users, n_products))
    max_gap = scoring_rule[0] - scoring_rule[-1]
    
    # Calculate number of users for each type
    type_counts = [int(n_users * ratio) for ratio in type_ratios]
    # Adjust the last type to ensure total count is exactly n_users
    type_counts[-1] = n_users - sum(type_counts[:-1])
    
    # Special case for two types to maintain backward compatibility
    if n_types == 2:
        first_products = [1, min(19, n_products-1)]
        
        # Define second-choice ranges for backward compatibility
        second_ranges = [
            (2, min(10, n_products)),          # Type 1: Choose from products 2-9
            (min(10, n_products), n_products)  # Type 2: Choose from products 10-18
        ]
    else:
        # Simple approach: first n_types products (except 0) are used for first place
        first_products = [i for i in range(1, n_types+1)]
        
        # Divide remaining products evenly for second choices
        remaining_products = [i for i in range(n_types+1, n_products)]
        products_per_type = max(1, len(remaining_products) // n_types)
        
        second_ranges = []
        for i in range(n_types):
            start_idx = i * products_per_type
            end_idx = start_idx + products_per_type
            
            if i == n_types - 1:  # Last type gets all remaining products
                end_idx = len(remaining_products)
                
            if start_idx < len(remaining_products):
                if end_idx <= len(remaining_products):
                    type_range = (remaining_products[start_idx], remaining_products[min(end_idx-1, len(remaining_products)-1)]+1)
                else:
                    type_range = (remaining_products[start_idx], n_products)
                second_ranges.append(type_range)
            else:
                # If we run out of products, reuse some from the beginning
                type_range = (n_types+1, n_types+2)
                second_ranges.append(type_range)
    
    start_idx = 0
    
    # Generate scores for each type of users
    for type_idx in range(n_types):
        n_type_users = type_counts[type_idx]
        end_idx = start_idx + n_type_users
        
        first_product = first_products[type_idx]
        
        for i in range(start_idx, end_idx):
            # Product 0 (target) always gets third place score
            third_place_score = 20
            user_scores[i, 0] = third_place_score
            
            # First ranked product for this type
            user_scores[i, first_product] = alpha * n_influencers * max_gap + third_place_score - 0.1
            
            # Choose second-ranked product from the specified range
            if type_idx < len(second_ranges):
                second_range = second_ranges[type_idx]
                available_indices = list(range(second_range[0], second_range[1]))
                
                # Remove first product if it's in the range
                if first_product in available_indices:
                    available_indices.remove(first_product)
                    
                if available_indices:  # Check if the range is valid
                    second_choice = np.random.choice(available_indices)
                    user_scores[i, second_choice] = third_place_score + beta * max_gap
                    
                    # Create list of remaining indices for random ranking
                    remaining_indices = list(range(1, n_products))
                    if first_product in remaining_indices:
                        remaining_indices.remove(first_product)
                    if second_choice in remaining_indices:
                        remaining_indices.remove(second_choice)
                else:
                    # If no valid indices for second place, use what's available
                    remaining_indices = list(range(1, n_products))
                    if first_product in remaining_indices:
                        remaining_indices.remove(first_product)
            else:
                # Fallback: no valid second range
                remaining_indices = list(range(1, n_products))
                if first_product in remaining_indices:
                    remaining_indices.remove(first_product)
            
            # Randomly assign scores to remaining positions
            np.random.shuffle(remaining_indices)
            for idx, pos in enumerate(remaining_indices):
                user_scores[i, pos] = idx * 0.01
        
        start_idx = end_idx
            
    return user_scores

def initialize_custom_scoring_rule(n_products):
    
    """
    Initialize scoring rule according to specified pattern:
    - Top position: 60
    - Second position: 49.9
    - Rest: (n_products-i)*0.001 where i is position from end
    """
    scoring_rule = np.zeros(n_products)
    scoring_rule[0] = 50      # Top position score
    scoring_rule[1] = 49.99  # Second position score
    scoring_rule[2] = 49.98
     
    # Fill remaining positions with small decreasing values
    for i in range(3, n_products):
        scoring_rule[i] = (n_products - i-1) * 0.001
        
    return scoring_rule


def initialize_network_optimized_scores(n_users, n_influencers, n_products, scoring_rule=None, seed=None, alpha=0.99, beta=0.9):
    """
    Initialize user scores by:
    1. Creating initial rankings (X) where c1 is ranked first, c0 is ranked third
    2. Converting rankings to scores (Y) using custom scoring pattern
    3. Solving for network weights W that minimize ||WX-Y|| with non-negative constraints
    4. Returning WX as the final user scores
    
    Args:
        n_users: Number of users
        n_influencers: Number of influencers (not used directly but kept for API consistency)
        n_products: Number of products
        scoring_rule: Not directly used but kept for API consistency
        seed: Random seed
        
    Returns:
        Array of user scores (n_users, n_products)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Initialize rankings X where c1 is first, c0 is third
    rankings = np.zeros((n_users, n_products), dtype=int)
    
    for i in range(n_users):
        # Set c1 as first ranked
        rankings[i, 1] = 0  # Rank 0 is the highest
        
        # Set c0 as third ranked
        rankings[i, 0] = 2  # Rank 2 is the third position
        
        # Randomly assign remaining ranks
        remaining_positions = list(range(n_products))
        remaining_positions.remove(0)  # Remove c0
        remaining_positions.remove(1)  # Remove c1
        
        remaining_ranks = list(range(n_products))
        remaining_ranks.remove(0)  # Remove first rank (assigned to c1)
        remaining_ranks.remove(2)  # Remove third rank (assigned to c0)
        
        np.random.shuffle(remaining_ranks)
        
        for pos, rank in zip(remaining_positions, remaining_ranks):
            rankings[i, pos] = rank
    
    # Step 2: Convert rankings to scores Y using custom scoring pattern
    initial_scores = np.zeros((n_users, n_products))
    max_gap = scoring_rule[0] - scoring_rule[-1]
    
    for i in range(n_users):
        # Set base score for third ranked item (c0)
        base_score = 20
        
        # Apply the mapping to each product based on its rank
        for j in range(n_products):
            if rankings[i, j] == 0:  # First ranked item (c1)
                initial_scores[i, j] = alpha * n_influencers * max_gap + base_score - 0.1
            elif rankings[i, j] == 2:  # Third ranked item (c0)
                initial_scores[i, j] = base_score
            elif rankings[i, j] == 1:  # Second ranked item
                initial_scores[i, j] = base_score + beta * max_gap
            else:  # All other items
                initial_scores[i, j] = (n_products - rankings[i, j]) * 0.01
    
    # Step 3: Solve for network weights W that minimize ||WX-Y||
    # We'll use scipy's non-negative least squares solver
    from scipy.optimize import nnls
    
    # Create X by applying scoring rule to rankings
    # Use advanced indexing to apply scoring rule to all rankings at once
    X = scoring_rule[rankings]
    
    Y = initial_scores  # We want to find W such that WX approximates Y
    
    # Initialize W matrix
    W = np.zeros((n_users, n_users))
    
    # Solve for each row of W^T (each column of W)
    for i in range(n_users):
        # Solve non-negative least squares for each user
        # Transpose X to match dimensions: X is (n_users, n_products), Y[i] is (n_products,)
        w_i, _ = nnls(X.T, Y[i])
        W[:, i] = w_i
    
    # Step 4: Return WX as final user scores
    # final_scores = W @ initial_scores
    final_scores = W @ X
    
    return final_scores, W, initial_scores

def lp_scoring_rule(n_products, p=1, convex=True, normalize=True):
    """
    Generate a scoring rule based on lp-norm curves. This generalizes the Borda count
    to allow for convex or concave scoring distributions.
    
    Parameters:
    -----------
    n_products : int
        Number of products (length of scoring rule)
    p : float
        The p-norm parameter (p=1 gives standard Borda count)
    convex : bool
        If True, generates convex scoring rule (more weight to top ranks)
        If False, generates concave scoring rule (more weight to lower ranks)
    normalize : bool
        If True, normalizes scores so max score = n_products-1 (like Borda)
        
    Returns:
    --------
    scoring_rule : numpy.ndarray
        Array of scores for each position from highest to lowest
    """
    # Generate the lp curve with many points for good interpolation
    x, y = lp_curve(convex=convex, p=p, square_size=1, num_points=1000)
    
    # Sort points by x value to ensure monotonicity
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    # Create interpolation function
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Generate positions (normalized to [0,1])
    positions = np.linspace(0, 1, n_products)
    
    # Get scores at these positions
    scores = f(positions)
    scores[0]=1.
    scores[1]=0.
    
    # For convex curve, we want highest score at position 0
    # For concave curve, we want highest score at position n_products-1
    # if convex:
    #     scores = scores[::-1]  # Reverse to get highest score first
    
    # Normalize if requested
    if normalize:
        max_score = n_products - 1
        scores = scores * max_score / np.max(scores)
        
    return scores

def plot_lp_scoring_rules(n_products=5, p_values=None, save_path=None):
    """
    Plot different LP-norm scoring rules for comparison
    
    Parameters:
    -----------
    n_products : int
        Number of products
    p_values : list of float
        List of p-values to plot
    save_path : str or None
        If provided, save the plot to this path
    """
    if p_values is None:
        p_values = [1, 1.5, 2, 4]
    
    plt.figure(figsize=(10, 6))
    positions = np.arange(n_products)
    
    # Plot Borda as reference
    borda = lp_scoring_rule(n_products, p=1, convex=True)
    plt.plot(positions, borda, 'k-', marker='o', label='Borda (p=1)')
    
    # Plot convex rules
    for p in p_values:
        if p == 1:  # Skip p=1 as it's already plotted as Borda
            continue
        scores = lp_scoring_rule(n_products, p=p, convex=True)
        plt.plot(positions, scores, marker='o', linestyle='-', label=f'Convex p={p}')
    
    # Plot concave rules
    for p in p_values:
        scores = lp_scoring_rule(n_products, p=p, convex=False)
        plt.plot(positions, scores, marker='s', linestyle='--', label=f'Concave p={p}')
    
    plt.xlabel('Rank Position')
    plt.ylabel('Score')
    plt.title('LP-Norm Scoring Rules')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def initialize_custom_network(n_users, n_influencers):
    """
    Initialize network weights where:
    - User-to-user weights are all zero
    - Influencer-to-user weights are all one
    """
    W = np.zeros((n_users + n_influencers, n_users))
    for i in range(n_users):
        W[i, i] = 1.0
    
    # Set influencer-to-user weights to 1
    W[n_users:, :] = 1.0
    
    return W


def initialize_powerlaw_scores(n_users, n_products, alpha=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    user_scores = np.zeros((n_users, n_products))
    positions = np.arange(n_products)
    scoring_rule = np.arange(n_products, 0, -1)
    
    # Calculate powerlaw probabilities (same distribution for both cases)
    probs = (positions + 1) ** alpha
    probs = probs / probs.sum()
    
    for i in range(n_users):
        # Assign score for index 0 (powerlaw favoring low scores)
        score_0 = np.random.choice(scoring_rule, p=probs)
        
        # Assign scores for last two indices (powerlaw favoring high scores)
        remaining_scores = list(set(scoring_rule) - {score_0})
        scores_high = np.random.choice(remaining_scores, size=2, 
                                     p=probs[:-1]/probs[:-1].sum(), 
                                     replace=False)
        
        # Randomly assign remaining scores to middle indices
        remaining_scores = list(set(scoring_rule) - {score_0} - set(scores_high))
        np.random.shuffle(remaining_scores)
        
        # Fill in scores array
        user_scores[i, 0] = score_0
        user_scores[i, -1] = scores_high[0]
        user_scores[i, -2] = scores_high[1]
        for j, score in enumerate(remaining_scores):
            user_scores[i, j+1] = score
            
    return user_scores

def initialize_network_weights(n_users, n_influencers, user_sparsity=0.3, inf_sparsity=0.7, inf_weight_multiplier=2.0, seed=None):
    """
    Initialize network weights with sparse connections and stronger influencer weights.
    
    Args:
        n_users: Number of users
        n_influencers: Number of influencers
        user_sparsity: Probability of zero weight between users
        inf_sparsity: Probability of zero weight from influencer to user
        inf_weight_multiplier: Multiplier for influencer weights vs user weights
        seed: Random seed
    
    Returns:
        Weight matrix W of shape (n_users + n_influencers, n_users)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize user-to-user weights
    W_users = np.random.random((n_users, n_users))
    user_mask = np.random.random((n_users, n_users)) < user_sparsity
    W_users[user_mask] = 0
    
    # Initialize influencer-to-user weights
    W_inf = np.random.random((n_influencers, n_users))
    inf_mask = np.random.random((n_influencers, n_users)) < inf_sparsity
    W_inf[inf_mask] = 0
    W_inf *= inf_weight_multiplier
    
    # Combine weights
    W = np.vstack([W_users, W_inf])
    
    return W

def run_comparison(n_users=10, n_influencers=5, n_products=4, alpha=2.0, 
                  user_sparsity=0.3, inf_sparsity=0.7, inf_weight_multiplier=2.0, seed=42):
    np.random.seed(seed)
    U = np.arange(n_users)
    C = np.arange(n_products)
    scoring_rule = np.arange(n_products, 0, -1)
    
    user_scores = initialize_powerlaw_scores(n_users, n_products, alpha, seed)
    W = initialize_network_weights(n_users, n_influencers, user_sparsity, 
                                 inf_sparsity, inf_weight_multiplier, seed)
    
    strategies = {
        'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'CARDRIM': cardrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
        'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
    }
    
    results = {}
    for name, rankings in strategies.items():
        agg_imp, card_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores)
        results[name] = {
            'AggRI': agg_imp, 
            'CardRI': card_imp,
            'Rankings': rankings,
            'Initial_Scores': user_scores,
            'Weights': W
        }
    
    
    return results

# def compute_max_score(U, W, C, k, scoring_rule, metric='agg'):
#     n_users = len(U)
#     n_products = len(C)
#     max_influence = W[n_users:n_users+k, :].sum(axis=0) * (scoring_rule[0] - scoring_rule[-1])
    
#     # Count users who can potentially improve
#     improvable_users = np.sum(max_influence > 0)
    
#     if metric == 'agg':
#         return improvable_users * (n_products - 1)
#     else:  # metric == 'card'
#         return improvable_users

def compute_max_score(U, W, C, k, user_scores, metric='agg'):
    """
    Compute the maximum possible ranking improvement for the target product (index 0).
    For each user, the maximum improvement is the difference between their initial
    ranking of the target product and the best possible ranking (1).
    
    Parameters:
        U: Array of user indices
        W: Weight matrix
        C: Array of product indices
        k: Number of influencers
        scoring_rule: Array of scores for each position
        user_scores: Initial user scores matrix
        metric: 'agg' or 'card'
    """
    n_users = len(U)
    
    # Calculate initial scores and rankings
    orig_scores = W[:n_users, :] @ user_scores
    orig_ranks = np.argsort(np.argsort(-orig_scores, axis=1), axis=1)
    
    # Get initial ranking of target product (index 0) for each user
    target_initial_ranks = orig_ranks[:, 0]
    
    # Maximum possible improvement is from current rank to rank 1
    max_improvements = target_initial_ranks
    
    if metric == 'agg':
        return np.sum(max_improvements)
    else:  # metric == 'card'
        return np.sum(max_improvements > 0)


def run_comparison_with_max(n_users=10, n_influencers=5, n_products=4, alpha=2.0, 
                          user_sparsity=0.3, inf_sparsity=0.7, inf_weight_multiplier=2.0, 
                          metric='agg', seed=42):
    np.random.seed(seed)
    U = np.arange(n_users)
    C = np.arange(n_products)
    scoring_rule = np.arange(n_products, 0, -1)
    
    user_scores = initialize_powerlaw_scores(n_users, n_products, alpha, seed)
    W = initialize_network_weights(n_users, n_influencers, user_sparsity, 
                                 inf_sparsity, inf_weight_multiplier, seed)
    
    max_score = compute_max_score(U, W, C, n_influencers, user_scores, metric)
    
    strategies = {
        'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'CARDRIM': cardrim(U, W, C, n_influencers, scoring_rule, user_scores),
        'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
        'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
    }
    
    results = {}
    for name, rankings in strategies.items():
        score = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, metric)
        results[name] = {
            'Score': score,
            'Rankings': rankings,
            'Ratio': score / max_score if max_score > 0 else 0
        }
    
    results['Maximum'] = {'Score': max_score}
    
    return results


def print_results(results, metric='agg'):
    print(f"\nResults Summary ({metric.upper()} metric):")
    print("=" * 40)
    max_score = results['Maximum']['Score']
    
    for strategy, data in results.items():
        if strategy != 'Maximum':
            print(f"\n{strategy}:")
            print(f"Score: {data['Score']:.2f}")
            print(f"Ratio to Max: {data['Ratio']:.2%}")
            print(f"First Influencer Ranking: {data['Rankings'][0]}")
    
    print(f"\nMaximum Possible Score: {max_score:.2f}")

# if __name__ == "__main__":
#     results = run_comparison_with_max(
#         n_users=10,
#         n_influencers=5,
#         n_products=4,
#         alpha=2.0,
#         user_sparsity=0.3,
#         inf_sparsity=0.7,
#         inf_weight_multiplier=2.0,
#         metric='agg'
#     )
#     print_results(results, 'agg')
def borda_rule(n_products):
    scoring_rule = np.arange(n_products, 0, -1)
    return scoring_rule


# if __name__ == "__main__":
    
#     # n_users=50
#     # n_influencers=20
#     # n_products=20
#     # user_sparsity=0.2  # 30% of user-user weights will be 0
#     # inf_sparsity=0.5   # 70% of influencer-user weights will be 0  
#     # inf_weight_multiplier=1.3 # Influencer weights 5x stronger
#     # seed=42
#     # alpha=10.
#     # metric='agg'
    
#     # np.random.seed(seed)
#     # U = np.arange(n_users)
#     # C = np.arange(n_products)

#     # # scoring_rule = np.arange(n_products, 0, -1)
#     # scoring_rule = border_rule(n_products)
    
#     # user_scores = initialize_powerlaw_scores(n_users, n_products, alpha, seed)
#     # W = initialize_network_weights(n_users, n_influencers, user_sparsity, 
#     #                              inf_sparsity, inf_weight_multiplier, seed)
    
#     # max_score = compute_max_score(U, W, C, n_influencers, scoring_rule, metric)
    
#     # strategies = {
#     #     'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores),
#     #     # 'CARDRIM': cardrim(U, W, C, n_influencers, scoring_rule, user_scores),
#     #     'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
#     #     'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores)
#     # }
    
#     # results = {}
#     # for name, rankings in strategies.items():
#     #     score = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, metric)
#     #     results[name] = {
#     #         'Score': score,
#     #         'Rankings': rankings,
#     #         'Ratio': score / max_score if max_score > 0 else 0
#     #     }
    
#     # results['Maximum'] = {'Score': max_score}
    
#     # # print("\nResults Summary:")
#     # # print("=" * 50)
#     # # for strategy, metrics in results.items():
#     # #     print(f"\n{strategy}:")
#     # #     # print(f"AggRI Score: {metrics['AggRI']}")
#     # #     print(f"CardRI Score: {metrics['CardRI']}")
#     # #     print(f"Example Ranking (First Influencer): {metrics['Rankings'][0]}")

#     # print_results(results, metric)

#     # Example usage
#     n_products = 21

#     # Generate standard Borda count scoring rule
#     borda = lp_scoring_rule(n_products, p=1, convex=True)
#     print("Borda count:", borda)
    
#     # Generate convex scoring rule (more weight to top ranks)
#     convex_rule = lp_scoring_rule(n_products, p=2, convex=True)
#     print("Convex rule (p=2):", convex_rule)
    
#     # Generate concave scoring rule (more weight to lower ranks)
#     concave_rule = lp_scoring_rule(n_products, p=2, convex=False)
#     print("Concave rule (p=2):", concave_rule)
    
#     # Plot and compare different rules
#     plot_lp_scoring_rules(n_products=n_products, p_values=[1, 1.5, 2, 4, float('inf')], save_path="lp_scoring_rules.png")

def run_custom_comparison(n_users=100, n_influencers=10, n_products=20, seed=42, alpha=10., beta=0.9, 
                         score_rule="borda", init_score="custom", convex=True, p=1, metric='agg'):
    """
    Run comparison with custom initialization parameters using specified metric
    """
    np.random.seed(seed)
    U = np.arange(n_users)
    C = np.arange(n_products)
    
    # Initialize scoring rule
    if score_rule == "borda":
        scoring_rule = borda_rule(n_products)
    elif score_rule == "custom":
        scoring_rule = initialize_custom_scoring_rule(n_products)
    elif score_rule == "lp":
        scoring_rule = lp_scoring_rule(n_products, p=p, convex=convex)
    else:
        raise ValueError("Invalid scoring rule type")
    
    # Initialize user scores
    if init_score == "custom":
        user_scores = initialize_custom_user_scores(n_users, n_influencers, n_products, scoring_rule, alpha, beta, seed)
    elif init_score == "two_type":
        user_scores = initialize_two_type_user_scores(n_users, n_influencers, n_products, scoring_rule, alpha, beta, seed=seed)
    elif init_score == "multi_type":
        user_scores = initialize_multi_type_user_scores(n_users, n_influencers, n_products, n_types=3, scoring_rule=scoring_rule, alpha=alpha, beta=beta, seed=seed)
    else:
        raise ValueError("Invalid user score initialization type")
    
    W = initialize_custom_network(n_users, n_influencers)
    
    # Use the specified metric for max score calculation
    max_score = compute_max_score(U, W, C, n_influencers, user_scores, metric)
    
    # All strategies use the same metric for consistency
    strategies = {
        'Random': random_strategy(U, W, C, n_influencers, scoring_rule, user_scores),
        'Independent': independent_strategy(U, W, C, n_influencers, scoring_rule, user_scores, metric),
        'Sequential': sequential_strategy(U, W, C, n_influencers, scoring_rule, user_scores, metric),
        'CAGGRIM': caggrim(U, W, C, n_influencers, scoring_rule, user_scores) if metric == 'agg' or metric == 'cagg' else cardrim(U, W, C, n_influencers, scoring_rule, user_scores)
    }
    
    results = {}
    for name, rankings in strategies.items():
        agg_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, 'agg')
        card_imp = evaluate_rankings(U, W, rankings, scoring_rule, user_scores, 'card')
        
        # Use the specified metric for ratio calculation
        selected_score = agg_imp if metric == 'agg' or metric == 'cagg' else card_imp
        
        results[name] = {
            'AggRI': agg_imp,
            'CardRI': card_imp,
            'Rankings': rankings,
            'Ratio': selected_score / max_score if max_score > 0 else 0
        }
    
    return results, user_scores, W, scoring_rule

def verify_cardrim(U, W, C, k, scoring_rule, user_scores, debug_level=1):
    """
    Run CARDRIM with verification checks and debug output
    
    Args:
        U, W, C, k, scoring_rule, user_scores: Same as cardrim
        debug_level: 0=minimal, 1=normal, 2=verbose
    
    Returns:
        Rankings from cardrim
    """
    n_users = len(U)
    n_products = len(C)
    M = calc_matrix(U, W, user_scores)
    
    if debug_level >= 1:
        print(f"Starting CARDRIM verification with {n_users} users, {n_products} products, {k} influencers")
        print(f"M shape: {M.shape}")
        print(f"Initial M non-zero elements: {np.count_nonzero(M)}")
    
    rankings = np.zeros((k, n_products), dtype=int)
    remaining_item_mask = np.ones(n_products, dtype=bool)
    remaining_item_mask[0] = False  # Target product is index 0
    remaining_user_mask = np.ones(n_users, dtype=bool)  # Track users who haven't been covered
    
    # Track how many users get covered at each position
    coverage_history = []
    
    for pos in range(n_products-1, 0, -1):
        if debug_level >= 1:
            print(f"\nPosition {pos}:")
            print(f"  Remaining users: {np.sum(remaining_user_mask)}/{n_users}")
            print(f"  Remaining items: {np.sum(remaining_item_mask)}/{n_products}")
        
        # Verify dimensions before calling greedy_card
        assert remaining_user_mask.shape == (n_users,), f"remaining_user_mask shape is {remaining_user_mask.shape}, expected {(n_users,)}"
        assert remaining_item_mask.shape == (n_products,), f"remaining_item_mask shape is {remaining_item_mask.shape}, expected {(n_products,)}"
        
        # Get M subset dimensions for verification
        M_subset = M[remaining_user_mask, :][:, remaining_item_mask]
        if debug_level >= 2:
            print(f"  M_subset shape: {M_subset.shape}")
        
        best_idx, covered_user_mask = greedy_card(U, W, M, remaining_item_mask, remaining_user_mask, pos+1, 
                                                 k, scoring_rule, n_users, tiebreak="random")
        
        # Verify covered_user_mask
        assert covered_user_mask.shape == (n_users,), f"covered_user_mask shape is {covered_user_mask.shape}, expected {(n_users,)}"
        
        # Verify best_idx is valid
        assert 0 <= best_idx < np.sum(remaining_item_mask), f"best_idx {best_idx} out of range [0, {np.sum(remaining_item_mask)-1}]"
        
        abs_idx = np.where(remaining_item_mask)[0][best_idx]
        rankings[:, pos] = abs_idx
        remaining_item_mask[abs_idx] = False
        
        # Count newly covered users
        newly_covered = remaining_user_mask & covered_user_mask
        num_newly_covered = np.sum(newly_covered)
        coverage_history.append(num_newly_covered)
        
        if debug_level >= 1:
            print(f"  Selected product: {abs_idx}")
            print(f"  Newly covered users: {num_newly_covered}")
        
        # Update which users have been covered
        old_remaining = np.sum(remaining_user_mask)
        remaining_user_mask = remaining_user_mask & ~covered_user_mask
        new_remaining = np.sum(remaining_user_mask)
        
        # Verify the update worked correctly
        assert old_remaining - new_remaining == num_newly_covered, \
            f"User count mismatch: {old_remaining} - {new_remaining} != {num_newly_covered}"
        
        # If all users are covered, we can stop
        if not np.any(remaining_user_mask):
            if debug_level >= 1:
                print(f"  All users covered! Filling remaining positions randomly.")
            
            # Fill remaining positions randomly
            for p in range(pos-1, 0, -1):
                avail_idx = np.random.choice(np.where(remaining_item_mask)[0])
                rankings[:, p] = avail_idx
                remaining_item_mask[avail_idx] = False
                if debug_level >= 2:
                    print(f"  Random fill position {p}: product {avail_idx}")
            break
    
    # Final verification
    if debug_level >= 1:
        print("\nFinal results:")
        print(f"  Total users covered: {n_users - np.sum(remaining_user_mask)}/{n_users}")
        print(f"  Coverage by position: {coverage_history}")
        print(f"  First influencer ranking: {rankings[0]}")
        
        # Verify no duplicates in rankings
        for i in range(k):
            unique_items = np.unique(rankings[i])
            assert len(unique_items) == n_products, f"Influencer {i} has duplicate products in ranking"
    
    return rankings

# Example usage:
# rankings = verify_cardrim(U, W, C, n_influencers, scoring_rule, user_scores, debug_level=1)