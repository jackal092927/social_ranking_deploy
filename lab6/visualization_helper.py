import numpy as np
import matplotlib.pyplot as plt
# import os
from scipy.interpolate import interp1d
import pandas as pd

def generate_summary_visualizations(all_results, all_ratios, alpha_values, all_beta_values, strategies, results_dir):
    """Generate summary visualizations across all alpha and beta values
    
    Args:
        all_results: Dictionary of results keyed by alpha, strategy, then beta index
        all_ratios: Dictionary of ratio results keyed by alpha, strategy, then beta index
        alpha_values: Array of alpha values tested
        all_beta_values: Dictionary of beta values for each alpha
        strategies: List of strategy names
        results_dir: Directory to save visualization files
        
    Returns:
        bool: True if all visualizations were created successfully
    """
    try:
        print("\nGenerating summary visualizations...")
        
        # Define common beta ranges for interpolation
        low_betas = np.linspace(0.11, 0.99, 30)
        high_betas = np.linspace(1.1, 9.5, 30)  # Use a reasonable upper limit
        
        for beta_range, beta_values in [("low", low_betas), ("high", high_betas)]:
            try:
                print(f"\nGenerating {beta_range} beta range visualizations...")
                
                # Create heatmaps for each strategy
                for strategy in strategies:
                    try:
                        # For each alpha, we need to interpolate the data onto a common beta grid
                        interpolated_data = []
                        
                        for i, alpha in enumerate(alpha_values):
                            # Extract original data points
                            original_betas = all_beta_values[alpha]
                            original_data = np.array(all_results[alpha][strategy])
                            
                            # Filter out NaN values
                            valid_mask = ~np.isnan(original_data)
                            valid_betas = original_betas[valid_mask]
                            valid_data = original_data[valid_mask]
                            
                            if len(valid_data) < 2:
                                # Not enough data points for interpolation
                                interpolated_data.append(np.zeros_like(beta_values))
                                continue
                            
                            # Filter to the relevant range
                            if beta_range == "low":
                                range_mask = valid_betas <= 1.0
                            else:  # high
                                range_mask = valid_betas >= 1.0
                                
                            range_betas = valid_betas[range_mask]
                            range_data = valid_data[range_mask]
                            
                            if len(range_data) < 2:
                                # Not enough data points in this range
                                interpolated_data.append(np.zeros_like(beta_values))
                                continue
                            
                            # Ensure beta values are within the range of original data
                            min_beta = max(beta_values[0], range_betas.min())
                            max_beta = min(beta_values[-1], range_betas.max())
                            
                            # Create mask for beta values in range
                            beta_mask = (beta_values >= min_beta) & (beta_values <= max_beta)
                            
                            # Initialize with zeros
                            interp_result = np.zeros_like(beta_values)
                            
                            # Only interpolate if we have a valid range
                            if np.any(beta_mask):
                                f = interp1d(range_betas, range_data, kind='linear', bounds_error=False, fill_value=np.nan)
                                interp_result[beta_mask] = f(beta_values[beta_mask])
                                
                            interpolated_data.append(interp_result)
                        
                        # Convert to numpy array
                        strategy_data = np.array(interpolated_data)
                        
                        # Create meshgrid for plotting
                        Alpha, Beta = np.meshgrid(alpha_values, beta_values, indexing='ij')
                        
                        # Only create plot if we have valid data
                        if not np.all(np.isnan(strategy_data)):
                            plt.figure(figsize=(10, 8))
                            
                            # Mask NaN values for plotting
                            masked_data = np.ma.masked_invalid(strategy_data)
                            
                            heatmap = plt.pcolormesh(Beta, Alpha, masked_data, cmap='viridis', shading='auto')
                            plt.colorbar(heatmap, label='AggRI Score')
                            plt.title(f'{strategy} Performance Across Alpha and {beta_range.capitalize()} Beta Values', fontsize=20)
                            plt.xlabel('Beta', fontsize=18)
                            plt.ylabel('Alpha', fontsize=18)
                            plt.tight_layout()
                            plt.savefig(f'{results_dir}/{strategy}_alpha_{beta_range}_beta_heatmap.png', dpi=300)
                            plt.close()
                            print(f"Created {beta_range} beta range heatmap for {strategy}")
                    except Exception as e:
                        print(f"Error creating {beta_range} beta range heatmap for {strategy}: {e}")
            except Exception as e:
                print(f"Error generating {beta_range} beta range visualizations: {e}")

        # Create individual plots for each alpha showing all strategies across the full beta range
        print("\nCreating individual alpha plots across full beta range...")
        for i, alpha in enumerate(alpha_values):
            try:
                beta_values = all_beta_values[alpha]
                
                plt.figure(figsize=(14, 8))
                for strategy in strategies:
                    data = np.array(all_results[alpha][strategy])
                    mask = np.isnan(data)
                    if not np.all(mask):  # Only plot if we have valid data
                        plt.plot(beta_values[~mask], data[~mask], marker='o', label=strategy)
                
                plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
                plt.title(f'Strategy Performance vs Beta (Alpha={alpha:.2f})', fontsize=20)
                plt.xlabel('Beta value', fontsize=18)
                plt.ylabel('AggRI Score', fontsize=18)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=18)
                plt.tight_layout()
                plt.savefig(f'{results_dir}/all_strategies_alpha_{alpha:.2f}.png', dpi=300)
                plt.close()
                print(f"Created full beta range plot for alpha={alpha:.2f}")
            except Exception as e:
                print(f"Error creating full beta range plot for alpha={alpha:.2f}: {e}")
                
        # Skip the best strategy map as it's difficult with different beta ranges per alpha
        print("\nSkipping best strategy map due to different beta ranges per alpha.")

        # Create a single plot showing CAGGRIM performance for all alphas
        try:
            plt.figure(figsize=(14, 10))
            
            for i, alpha in enumerate(alpha_values):
                beta_values = all_beta_values[alpha]
                data = np.array(all_results[alpha]['CAGGRIM'])
                mask = np.isnan(data)
                
                if not np.all(mask):  # Only plot if we have valid data
                    plt.plot(beta_values[~mask], data[~mask], marker='o', label=f'Alpha={alpha:.2f}')
            
            plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
            plt.title('CAGGRIM Performance Across All Alpha and Beta Values', fontsize=20)
            plt.xlabel('Beta value', fontsize=18)
            plt.ylabel('AggRI Score', fontsize=18)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/CAGGRIM_all_alphas.png', dpi=300)
            plt.close()
            print("Created CAGGRIM performance plot for all alphas")
        except Exception as e:
            print(f"Error creating CAGGRIM performance plot for all alphas: {e}")
        
        # Create plots showing performance for all alphas for other strategies
        for strategy in ['Random', 'Sequential', 'Independent']:
            try:
                plt.figure(figsize=(14, 10))
                
                for i, alpha in enumerate(alpha_values):
                    beta_values = all_beta_values[alpha]
                    data = np.array(all_results[alpha][strategy])
                    mask = np.isnan(data)
                    
                    if not np.all(mask):  # Only plot if we have valid data
                        plt.plot(beta_values[~mask], data[~mask], marker='o', label=f'Alpha={alpha:.2f}')
                
                plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
                plt.title(f'{strategy} Performance Across All Alpha and Beta Values', fontsize=20)
                plt.xlabel('Beta value', fontsize=18)
                plt.ylabel('AggRI Score', fontsize=18)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=18)
                plt.tight_layout()
                plt.savefig(f'{results_dir}/{strategy}_all_alphas.png', dpi=300)
                plt.close()
                print(f"Created {strategy} performance plot for all alphas")
            except Exception as e:
                print(f"Error creating {strategy} performance plot for all alphas: {e}")
        
        # Skip 3D surface plot as it's difficult with different beta ranges per alpha
        print("Skipping 3D surface plot due to different beta ranges per alpha.")

        # Save all results to a CSV file for later analysis
        try:
            # Create a DataFrame to store all results
            results_df = []
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                for beta_idx, beta in enumerate(beta_values):
                    if beta_idx >= len(all_results[alpha][strategies[0]]):
                        continue
                    row = {'Alpha': alpha, 'Beta': beta}
                    for strategy in strategies:
                        if beta_idx < len(all_results[alpha][strategy]):
                            row[f'{strategy}_Score'] = all_results[alpha][strategy][beta_idx]
                            row[f'{strategy}_Ratio'] = all_ratios[alpha][strategy][beta_idx]
                        else:
                            row[f'{strategy}_Score'] = np.nan
                            row[f'{strategy}_Ratio'] = np.nan
                    results_df.append(row)
            
            results_df = pd.DataFrame(results_df)
            results_df.to_csv(f'{results_dir}/alpha_beta_sensitivity_results.csv', index=False)
            print(f"Results saved to {results_dir}/alpha_beta_sensitivity_results.csv")
        except Exception as e:
            print(f"Error saving CSV: {e}")
        
        # Create a summary table with max scores for each strategy at each alpha
        try:
            summary_data = []
            for alpha in alpha_values:
                row = {'Alpha': alpha}
                for strategy in strategies:
                    scores = np.array(all_results[alpha][strategy])
                    valid_scores = scores[~np.isnan(scores)]
                    if len(valid_scores) > 0:
                        max_score = np.max(valid_scores)
                        max_beta_idx = np.nanargmax(scores)
                        max_beta = all_beta_values[alpha][max_beta_idx]
                        row[f'{strategy}_Max_Score'] = max_score
                        row[f'{strategy}_Optimal_Beta'] = max_beta
                    else:
                        row[f'{strategy}_Max_Score'] = np.nan
                        row[f'{strategy}_Optimal_Beta'] = np.nan
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f'{results_dir}/optimal_beta_summary.csv', index=False)
            print(f"Summary of optimal beta values saved to {results_dir}/optimal_beta_summary.csv")
        except Exception as e:
            print(f"Error creating summary table: {e}")
        
        return True
    except Exception as e:
        print(f"Error in summary visualization generation: {e}")
        return False
    

def create_relative_performance_visualizations(all_results, all_ratios, alpha_values, all_beta_values, strategies, results_dir):
    """Create visualizations that compare the performance of strategies relative to CAGGRIM.
    
    Args:
        all_results: Dictionary of results keyed by alpha, strategy, then beta index
        all_ratios: Dictionary of ratio results keyed by alpha, strategy, then beta index
        alpha_values: Array of alpha values tested
        all_beta_values: Dictionary of beta values for each alpha
        strategies: List of strategy names
        results_dir: Directory to save visualization files
        
    Returns:
        bool: True if all visualizations were created successfully
    """
    try:
        print("\nGenerating relative performance visualizations...")
        
        # Create relative performance heatmaps for each strategy compared to CAGGRIM
        create_relative_heatmaps(all_results, alpha_values, all_beta_values, strategies, results_dir)
        
        # Create line plots for each alpha showing relative performance across beta values
        create_alpha_relative_line_plots(all_results, alpha_values, all_beta_values, strategies, results_dir)
        
        # Create victory maps showing where each strategy outperforms CAGGRIM
        create_victory_maps(all_results, alpha_values, all_beta_values, strategies, results_dir)
        
        # Create summary plots with different statistics across all alphas for each strategy
        create_strategy_mean_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir)
        create_strategy_median_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir)
        create_strategy_max_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir)
        create_strategy_min_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir)
        
        return True
    except Exception as e:
        print(f"Error in relative performance visualization generation: {e}")
        return False

def create_relative_heatmaps(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create heatmaps showing relative performance compared to CAGGRIM for low and high beta ranges."""
    # Define common beta ranges for interpolation
    low_betas = np.linspace(0.11, 0.99, 30)
    high_betas = np.linspace(1.1, 9.5, 30)  # Use a reasonable upper limit
    
    for beta_range, beta_values in [("low", low_betas), ("high", high_betas)]:
        try:
            print(f"\nGenerating {beta_range} beta range relative performance heatmaps...")
            
            # Create heatmaps for each strategy
            for strategy in strategies:
                if strategy != 'CAGGRIM':  # Skip CAGGRIM itself
                    try:
                        # For each alpha, calculate relative performance compared to CAGGRIM
                        relative_data = []
                        
                        for i, alpha in enumerate(alpha_values):
                            # Extract original data points for this strategy and CAGGRIM
                            original_betas = all_beta_values[alpha]
                            strategy_scores = np.array(all_results[alpha][strategy])
                            caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                            
                            # Calculate relative performance where both have valid data
                            valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                            valid_betas = original_betas[valid_mask]
                            valid_relative = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                            
                            if len(valid_relative) < 2:
                                # Not enough data points for interpolation
                                relative_data.append(np.zeros_like(beta_values))
                                continue
                            
                            # Filter to the relevant range
                            if beta_range == "low":
                                range_mask = valid_betas <= 1.0
                            else:  # high
                                range_mask = valid_betas >= 1.0
                                
                            range_betas = valid_betas[range_mask]
                            range_relative = valid_relative[range_mask]
                            
                            if len(range_relative) < 2:
                                # Not enough data points in this range
                                relative_data.append(np.zeros_like(beta_values))
                                continue
                            
                            # Ensure beta values are within the range of original data
                            min_beta = max(beta_values[0], range_betas.min())
                            max_beta = min(beta_values[-1], range_betas.max())
                            
                            # Create mask for beta values in range
                            beta_mask = (beta_values >= min_beta) & (beta_values <= max_beta)
                            
                            # Initialize with zeros
                            interp_result = np.zeros_like(beta_values)
                            
                            # Only interpolate if we have a valid range
                            if np.any(beta_mask):
                                f = interp1d(range_betas, range_relative, kind='linear', bounds_error=False, fill_value=np.nan)
                                interp_result[beta_mask] = f(beta_values[beta_mask])
                                
                            relative_data.append(interp_result)
                        
                        # Convert to numpy array
                        relative_data = np.array(relative_data)
                        
                        # Create meshgrid for plotting
                        Alpha, Beta = np.meshgrid(alpha_values, beta_values, indexing='ij')
                        
                        # Only create plot if we have valid data
                        if not np.all(np.isnan(relative_data)):
                            plt.figure(figsize=(10, 8))
                            
                            # Mask NaN values for plotting
                            masked_data = np.ma.masked_invalid(relative_data)
                            
                            # Use a diverging colormap centered at 1.0 (equal performance)
                            # Values > 1 mean the strategy outperforms CAGGRIM
                            # Values < 1 mean CAGGRIM outperforms the strategy
                            vmin = max(0, np.nanmin(masked_data))
                            vmax = min(2, np.nanmax(masked_data))
                            
                            # Make the colormap symmetrical around 1 if possible
                            if vmin < 1 and vmax > 1:
                                vmin = min(vmin, 2-vmax)
                                vmax = max(vmax, 2-vmin)
                            
                            heatmap = plt.pcolormesh(Beta, Alpha, masked_data, cmap='RdYlGn', 
                                                    vmin=vmin, vmax=vmax, shading='auto')
                            colorbar = plt.colorbar(heatmap)
                            colorbar.set_label(f'{strategy}/CAGGRIM Ratio')
                            
                            # Add contour line at 1.0 (equal performance)
                            contour = plt.contour(Beta, Alpha, masked_data, levels=[1.0], 
                                                colors='black', linewidths=2, linestyles='dashed')
                            plt.clabel(contour, inline=True, fontsize=10, fmt='%.1f')
                            
                            plt.title(f'{strategy} vs CAGGRIM Performance Ratio ({beta_range.capitalize()} Beta Range)', fontsize=20)
                            plt.xlabel('Beta', fontsize=18)
                            plt.ylabel('Alpha', fontsize=18)
                            plt.tight_layout()
                            plt.savefig(f'{results_dir}/{strategy}_vs_CAGGRIM_ratio_{beta_range}.png', dpi=300)
                            plt.close()
                            print(f"Created {beta_range} beta range relative performance heatmap for {strategy}")
                    except Exception as e:
                        print(f"Error creating {beta_range} beta range relative performance heatmap for {strategy}: {e}")
        except Exception as e:
            print(f"Error generating {beta_range} beta range relative performance heatmaps: {e}")

def create_alpha_relative_line_plots(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create line plots for each alpha showing relative performance of strategies vs CAGGRIM."""
    try:
        print("\nCreating alpha-specific relative performance line plots...")
        
        # For each alpha, create a plot showing relative performance of all strategies vs CAGGRIM
        for alpha in alpha_values:
            plt.figure(figsize=(14, 8))
            beta_values = all_beta_values[alpha]
            
            for strategy in strategies:
                if strategy == 'CAGGRIM':
                    continue  # Skip CAGGRIM itself
                    
                # Calculate relative performance
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Mask where either has NaN or CAGGRIM is zero
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = beta_values[valid_mask]
                valid_relative = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                if len(valid_relative) > 0:
                    plt.plot(valid_betas, valid_relative, marker='o', label=strategy)
            
            # Add reference line at 1.0 (equal to CAGGRIM)
            plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
            
            # Add vertical line at beta=1.0
            plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
            
            plt.title(f'Strategy Performance Relative to CAGGRIM (Alpha={alpha:.2f})', fontsize=20)
            plt.xlabel('Beta value', fontsize=18)
            plt.ylabel('Performance Ratio (Strategy/CAGGRIM)', fontsize=18)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=18)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/relative_performance_alpha_{alpha:.2f}.png', dpi=300)
            plt.close()
            print(f"Created relative performance plot for alpha={alpha:.2f}")
    except Exception as e:
        print(f"Error creating alpha-specific relative performance line plots: {e}")

def create_victory_maps(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create maps showing regions where each strategy outperforms CAGGRIM."""
    try:
        print("\nCreating strategy victory maps...")
        
        # For each strategy (except CAGGRIM), create a "victory map"
        for strategy in strategies:
            if strategy == 'CAGGRIM':
                continue
                
            plt.figure(figsize=(12, 8))
            
            # We'll create a single visualization after interpolating to a common grid
            common_alphas = alpha_values
            common_betas = np.linspace(0.2, 8.0, 50)  # Use a wide range
            
            Alpha, Beta = np.meshgrid(common_alphas, common_betas, indexing='ij')
            victory_map = np.zeros(Alpha.shape)
            
            # For each alpha, determine where this strategy beats CAGGRIM
            for i, alpha in enumerate(common_alphas):
                original_betas = all_beta_values[alpha]
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Calculate relative performance
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = original_betas[valid_mask]
                valid_relative = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                if len(valid_relative) < 2:
                    continue
                    
                # Interpolate to common beta grid
                # Ensure we're within the valid range
                interp_mask = (common_betas >= valid_betas.min()) & (common_betas <= valid_betas.max())
                
                if np.any(interp_mask):
                    f = interp1d(valid_betas, valid_relative, kind='linear', bounds_error=False, fill_value=np.nan)
                    victory_map[i, interp_mask] = f(common_betas[interp_mask])
            
            # Mask values where strategy doesn't beat CAGGRIM (ratio <= 1)
            victory_regions = np.ma.masked_where(victory_map <= 1.0, victory_map)
            
            # Color the regions where this strategy beats CAGGRIM
            plt.pcolormesh(Beta, Alpha, victory_regions, cmap='RdYlGn', vmin=1.0, vmax=1.5, shading='auto')
            plt.colorbar(label=f'{strategy}/CAGGRIM Ratio')
            
            # Add contour at ratio=1.0 to show boundary
            contour = plt.contour(Beta, Alpha, np.ma.masked_invalid(victory_map), 
                                levels=[1.0], colors='black', linewidths=2)
            plt.clabel(contour, inline=True, fontsize=10, fmt='%.1f')
            
            # Add a fill for regions where the strategy is worse than CAGGRIM
            worse_regions = np.ma.masked_where((victory_map > 1.0) | np.isnan(victory_map), np.ones_like(victory_map))
            plt.pcolormesh(Beta, Alpha, worse_regions, cmap='gray', alpha=0.3, shading='auto')
            
            plt.title(f'Regions where {strategy} Outperforms CAGGRIM', fontsize=20)
            plt.xlabel('Beta', fontsize=18)
            plt.ylabel('Alpha', fontsize=18)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/{strategy}_victory_map.png', dpi=300)
            plt.close()
            print(f"Created victory map for {strategy}")
    except Exception as e:
        print(f"Error creating victory maps: {e}")

def create_strategy_mean_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create a summary plot showing mean performance across all alphas for each strategy.
    
    For each strategy, plot a curve where:
    - X-axis: beta values
    - Y-axis: mean score across all alphas for each beta value
    """
    try:
        print("\nCreating strategy mean performance plot across all alphas...")
        
        plt.figure(figsize=(14, 8))
        
        # Define a common beta grid for visualization
        common_betas = np.linspace(0.2, 8.0, 100)
        
        # For each strategy, compute and plot the mean performance across all alphas
        for strategy in strategies:
            # Dictionary to collect scores for each beta value
            beta_scores = {}
            
            # Collect all scores for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                scores = np.array(all_results[alpha][strategy])
                
                for i, beta in enumerate(beta_values):
                    if i < len(scores) and not np.isnan(scores[i]):
                        if beta not in beta_scores:
                            beta_scores[beta] = []
                        beta_scores[beta].append(scores[i])
            
            # Calculate mean score for each beta value
            beta_list = sorted(beta_scores.keys())
            mean_scores = [np.mean(beta_scores[beta]) for beta in beta_list]
            
            # Plot the mean scores
            plt.plot(beta_list, mean_scores, marker='o', label=strategy, linewidth=2)
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Mean Strategy Performance Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Mean AggRI Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_mean_performance.png', dpi=300)
        plt.close()
        
        # Also create a version with ratio to CAGGRIM
        plt.figure(figsize=(14, 8))
        
        # For each strategy (except CAGGRIM), compute and plot the mean ratio to CAGGRIM
        for strategy in strategies:
            if strategy == 'CAGGRIM':
                continue
                
            # Dictionary to collect ratios for each beta value
            beta_ratios = {}
            
            # Collect all ratios for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Calculate ratios where both have valid data
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = beta_values[valid_mask]
                valid_ratios = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                for i, beta in enumerate(valid_betas):
                    if i < len(valid_ratios):
                        if beta not in beta_ratios:
                            beta_ratios[beta] = []
                        beta_ratios[beta].append(valid_ratios[i])
            
            # Calculate mean ratio for each beta value
            beta_list = sorted(beta_ratios.keys())
            mean_ratios = [np.mean(beta_ratios[beta]) for beta in beta_list]
            
            # Plot the mean ratios
            plt.plot(beta_list, mean_ratios, marker='o', label=strategy, linewidth=2)
        
        # Add horizontal line at ratio=1.0 (equal to CAGGRIM)
        plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Mean Strategy Performance Ratio to CAGGRIM Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Mean Performance Ratio (Strategy/CAGGRIM)', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_mean_ratio_to_caggrim.png', dpi=300)
        plt.close()
        
        print("Created strategy mean performance plots across all alphas")
    except Exception as e:
        print(f"Error creating strategy mean performance plot: {e}")

def create_strategy_median_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create a summary plot showing median performance across all alphas for each strategy.
    
    For each strategy, plot a curve where:
    - X-axis: beta values
    - Y-axis: median score across all alphas for each beta value
    """
    try:
        print("\nCreating strategy median performance plot across all alphas...")
        
        plt.figure(figsize=(14, 8))
        
        # For each strategy, compute and plot the median performance across all alphas
        for strategy in strategies:
            # Dictionary to collect scores for each beta value
            beta_scores = {}
            
            # Collect all scores for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                scores = np.array(all_results[alpha][strategy])
                
                for i, beta in enumerate(beta_values):
                    if i < len(scores) and not np.isnan(scores[i]):
                        if beta not in beta_scores:
                            beta_scores[beta] = []
                        beta_scores[beta].append(scores[i])
            
            # Calculate median score for each beta value
            beta_list = sorted(beta_scores.keys())
            median_scores = [np.median(beta_scores[beta]) for beta in beta_list]
            
            # Plot the median scores
            plt.plot(beta_list, median_scores, marker='o', label=strategy, linewidth=2)
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Median Strategy Performance Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Median AggRI Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_median_performance.png', dpi=300)
        plt.close()
        
        # Also create a version with ratio to CAGGRIM
        plt.figure(figsize=(14, 8))
        
        # For each strategy (except CAGGRIM), compute and plot the median ratio to CAGGRIM
        for strategy in strategies:
            if strategy == 'CAGGRIM':
                continue
                
            # Dictionary to collect ratios for each beta value
            beta_ratios = {}
            
            # Collect all ratios for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Calculate ratios where both have valid data
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = beta_values[valid_mask]
                valid_ratios = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                for i, beta in enumerate(valid_betas):
                    if i < len(valid_ratios):
                        if beta not in beta_ratios:
                            beta_ratios[beta] = []
                        beta_ratios[beta].append(valid_ratios[i])
            
            # Calculate median ratio for each beta value
            beta_list = sorted(beta_ratios.keys())
            median_ratios = [np.median(beta_ratios[beta]) for beta in beta_list]
            
            # Plot the median ratios
            plt.plot(beta_list, median_ratios, marker='o', label=strategy, linewidth=2)
        
        # Add horizontal line at ratio=1.0 (equal to CAGGRIM)
        plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Median Strategy Performance Ratio to CAGGRIM Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Median Performance Ratio (Strategy/CAGGRIM)', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_median_ratio_to_caggrim.png', dpi=300)
        plt.close()
        
        print("Created strategy median performance plots across all alphas")
    except Exception as e:
        print(f"Error creating strategy median performance plot: {e}")

def create_strategy_max_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create a summary plot showing maximum performance across all alphas for each strategy.
    
    For each strategy, plot a curve where:
    - X-axis: beta values
    - Y-axis: maximum score across all alphas for each beta value
    """
    try:
        print("\nCreating strategy maximum performance plot across all alphas...")
        
        plt.figure(figsize=(14, 8))
        
        # For each strategy, compute and plot the maximum performance across all alphas
        for strategy in strategies:
            # Dictionary to collect scores for each beta value
            beta_scores = {}
            
            # Collect all scores for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                scores = np.array(all_results[alpha][strategy])
                
                for i, beta in enumerate(beta_values):
                    if i < len(scores) and not np.isnan(scores[i]):
                        if beta not in beta_scores:
                            beta_scores[beta] = []
                        beta_scores[beta].append(scores[i])
            
            # Calculate max score for each beta value
            beta_list = sorted(beta_scores.keys())
            max_scores = [np.max(beta_scores[beta]) for beta in beta_list]
            
            # Plot the max scores
            plt.plot(beta_list, max_scores, marker='o', label=strategy, linewidth=2)
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Maximum Strategy Performance Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Maximum AggRI Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_max_performance.png', dpi=300)
        plt.close()
        
        # Also create a version with ratio to CAGGRIM
        plt.figure(figsize=(14, 8))
        
        # For each strategy (except CAGGRIM), compute and plot the max ratio to CAGGRIM
        for strategy in strategies:
            if strategy == 'CAGGRIM':
                continue
                
            # Dictionary to collect ratios for each beta value
            beta_ratios = {}
            
            # Collect all ratios for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Calculate ratios where both have valid data
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = beta_values[valid_mask]
                valid_ratios = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                for i, beta in enumerate(valid_betas):
                    if i < len(valid_ratios):
                        if beta not in beta_ratios:
                            beta_ratios[beta] = []
                        beta_ratios[beta].append(valid_ratios[i])
            
            # Calculate max ratio for each beta value
            beta_list = sorted(beta_ratios.keys())
            max_ratios = [np.max(beta_ratios[beta]) for beta in beta_list]
            
            # Plot the max ratios
            plt.plot(beta_list, max_ratios, marker='o', label=strategy, linewidth=2)
        
        # Add horizontal line at ratio=1.0 (equal to CAGGRIM)
        plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Maximum Strategy Performance Ratio to CAGGRIM Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Maximum Performance Ratio (Strategy/CAGGRIM)', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_max_ratio_to_caggrim.png', dpi=300)
        plt.close()
        
        print("Created strategy maximum performance plots across all alphas")
    except Exception as e:
        print(f"Error creating strategy maximum performance plot: {e}")

def create_strategy_min_performance_plot(all_results, alpha_values, all_beta_values, strategies, results_dir):
    """Create a summary plot showing minimum performance across all alphas for each strategy.
    
    For each strategy, plot a curve where:
    - X-axis: beta values
    - Y-axis: minimum score across all alphas for each beta value
    """
    try:
        print("\nCreating strategy minimum performance plot across all alphas...")
        
        plt.figure(figsize=(14, 8))
        
        # For each strategy, compute and plot the minimum performance across all alphas
        for strategy in strategies:
            # Dictionary to collect scores for each beta value
            beta_scores = {}
            
            # Collect all scores for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                scores = np.array(all_results[alpha][strategy])
                
                for i, beta in enumerate(beta_values):
                    if i < len(scores) and not np.isnan(scores[i]):
                        if beta not in beta_scores:
                            beta_scores[beta] = []
                        beta_scores[beta].append(scores[i])
            
            # Calculate min score for each beta value
            beta_list = sorted(beta_scores.keys())
            min_scores = [np.min(beta_scores[beta]) for beta in beta_list]
            
            # Plot the min scores
            plt.plot(beta_list, min_scores, marker='o', label=strategy, linewidth=2)
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Minimum Strategy Performance Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Minimum AggRI Score', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_min_performance.png', dpi=300)
        plt.close()
        
        # Also create a version with ratio to CAGGRIM
        plt.figure(figsize=(14, 8))
        
        # For each strategy (except CAGGRIM), compute and plot the min ratio to CAGGRIM
        for strategy in strategies:
            if strategy == 'CAGGRIM':
                continue
                
            # Dictionary to collect ratios for each beta value
            beta_ratios = {}
            
            # Collect all ratios for each beta value across all alphas
            for alpha in alpha_values:
                beta_values = all_beta_values[alpha]
                strategy_scores = np.array(all_results[alpha][strategy])
                caggrim_scores = np.array(all_results[alpha]['CAGGRIM'])
                
                # Calculate ratios where both have valid data
                valid_mask = ~np.isnan(strategy_scores) & ~np.isnan(caggrim_scores) & (caggrim_scores != 0)
                valid_betas = beta_values[valid_mask]
                valid_ratios = strategy_scores[valid_mask] / caggrim_scores[valid_mask]
                
                for i, beta in enumerate(valid_betas):
                    if i < len(valid_ratios):
                        if beta not in beta_ratios:
                            beta_ratios[beta] = []
                        beta_ratios[beta].append(valid_ratios[i])
            
            # Calculate min ratio for each beta value
            beta_list = sorted(beta_ratios.keys())
            min_ratios = [np.min(beta_ratios[beta]) for beta in beta_list]
            
            # Plot the min ratios
            plt.plot(beta_list, min_ratios, marker='o', label=strategy, linewidth=2)
        
        # Add horizontal line at ratio=1.0 (equal to CAGGRIM)
        plt.axhline(y=1.0, color='r', linestyle='--', label='CAGGRIM baseline')
        
        # Add vertical line at beta=1.0
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.title('Minimum Strategy Performance Ratio to CAGGRIM Across All Alpha Values', fontsize=20)
        plt.xlabel('Beta value', fontsize=18)
        plt.ylabel('Minimum Performance Ratio (Strategy/CAGGRIM)', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        plt.xscale('log')  # Use log scale for better visualization of beta range
        plt.tight_layout()
        plt.savefig(f'{results_dir}/allstrategies_min_ratio_to_caggrim.png', dpi=300)
        plt.close()
        
        print("Created strategy minimum performance plots across all alphas")
    except Exception as e:
        print(f"Error creating strategy minimum performance plot: {e}")
