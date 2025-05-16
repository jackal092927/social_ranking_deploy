import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import datetime

def plot_multiple_methods(methods=['CAGGRIM', 'Sequential', 'Independent'], save_prefix='methods_comparison', 
                          filepath='lab6/results/lab6exp1-20250514-224014__agg__borda__custom-init/alpha_beta_sensitivity_results.csv',
                          save_dir='.'):
    """
    Plot performance of multiple methods together on the same 3D surface.
    
    Parameters:
    -----------
    methods : list
        List of methods to plot. Options: 'CAGGRIM', 'Sequential', 'Independent', 'Random'
    save_prefix : str
        Prefix for the saved filename
    filepath : str
        Path to the CSV file containing the data
    save_dir : str
        Directory to save the plot in
    """
    # Read data from CSV file
    data = pd.read_csv(filepath)

    # Map method name to corresponding column names
    method_columns = {
        'CAGGRIM': 'CAGGRIM_Score',
        'Sequential': 'Sequential_Score',
        'Independent': 'Independent_Score',
        'Random': 'Random_Score'
    }
    
    # Extract unique alpha and beta values
    alphas = sorted(data['Alpha'].unique())
    betas = sorted(data['Beta'].unique())

    # Create a grid for plotting
    beta_mesh, alpha_mesh = np.meshgrid(betas, alphas)
    
    # Create 3D figure with specific size
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define distinct colors for different methods - using non-reversed colormaps
    method_colors = {
        'CAGGRIM': plt.cm.Reds,    # Red gradient for CAGGRIM (switched)
        'Sequential': plt.cm.Greens,  # Green gradient for Sequential (switched)
        'Independent': plt.cm.autumn,
        'Random': plt.cm.Blues     # Blue gradient (darker=higher)
    }
    
    # Create a global color bar to show the value scale (for all methods)
    # Create a mock surface for the global colorbar
    min_value = 0
    max_value = 200
    norm = plt.Normalize(min_value, max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Empty array for the scalar mappable
    
    # Plot each method with a different color
    for i, method in enumerate(methods):
        # Get the appropriate column name for the selected method
        score_column = method_columns.get(method, 'CAGGRIM_Score')
        
        # Initialize the score grid with NaN values
        score_grid = np.full(alpha_mesh.shape, np.nan)
        
        # Fill in the score grid with method score values
        for j, alpha in enumerate(alphas):
            for k, beta in enumerate(betas):
                row = data[(data['Alpha'] == alpha) & (data['Beta'] == beta)]
                if not row.empty:
                    score_grid[j, k] = row[score_column].values[0]
        
        # Calculate method-specific min/max for better color distribution
        valid_scores = score_grid[~np.isnan(score_grid)]
        if len(valid_scores) > 0:
            # Adjust minimum values to avoid too light colors, more aggressively for Sequential and Random
            if method == 'CAGGRIM':
                method_min = max(0, np.min(valid_scores) * 0.8)
            else:
                # For Sequential and Random, use a higher minimum to make light areas darker
                method_min = max(0, np.min(valid_scores) * 0.5)   # More aggressive scaling (was 0.5)
            method_max = np.max(valid_scores)
        else:
            method_min = 0
            method_max = 200
            
        # Plot the surface with a specific colormap and adjusted range for each method
        surf = ax.plot_surface(beta_mesh, alpha_mesh, score_grid, 
                              cmap=method_colors[method],
                              vmin=method_min, vmax=method_max,  # Method-specific range
                              linewidth=0,
                              antialiased=True,
                              edgecolor='none',
                              alpha=0.7,  # Add some transparency for overlapping surfaces
                              label=method)  # Add label for legend
    
    # Create proxy artists for the legend (workaround for 3D surface plots)
    proxy_artists = []
    for method in methods:
        color = method_colors[method](0.7)  # Get darker color from colormap (0.7 instead of 0.3)
        proxy = plt.Rectangle((0, 0), 1, 1, fc=color)
        proxy_artists.append(proxy)
    
    # Add legend to the bottom center above the x-y intersection
    ax.legend(proxy_artists, methods, loc='lower center', fontsize=16, bbox_to_anchor=(0.5, 0.1))
    
    # Remove the color bar
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    # cbar.ax.tick_params(labelsize=16)
    # cbar.set_label('AggRI Score (Darker→Lower, Lighter→Higher)', fontsize=18, labelpad=15)

    # Set axis labels with large font size (no LaTeX)
    ax.set_xlabel('Beta', fontsize=22, labelpad=15)
    ax.set_ylabel('Alpha', fontsize=22, labelpad=25)  # Kept the increased labelpad
    
    # Remove default z-label
    # ax.set_zlabel('AggRI', fontsize=24, labelpad=20)
    
    # Manually place z-label at the top of z-axis
    ax.text(10, 1.0, 200, 'AggRI', fontsize=28, ha='center', va='bottom')
    
    # Adjust z-label position for better visibility
    # ax.zaxis._axinfo['label']['space_factor'] = 3.0

    # Adjust viewing angle to match reference image
    ax.view_init(elev=25, azim=-45)

    # Flip the axis direction to match the reference image
    ax.invert_xaxis()

    # Add grid lines to make it easier to read values
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)

    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 1.0)
    ax.set_zlim(0, 200)

    # Customize tick labels to be larger
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)

    # Ensure the plot fills the figure
    plt.tight_layout(pad=2.5)  # Increase padding to make room for labels

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure with high resolution and more padding
    methods_str = '_'.join([m.lower() for m in methods])
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'{save_prefix}_{methods_str}_comparison_{timestamp}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)  # Increased padding
    print(f"Plot saved as {save_path}")

    # Show plot
    plt.show()

def plot_method_performance(method='CAGGRIM', save_prefix='method_performance', 
                           filepath='lab6/results/lab6exp1-20250514-224014__agg__borda__custom-init/alpha_beta_sensitivity_results.csv',
                           save_dir='.'):
    """
    Plot performance of a single method across alpha and beta values.
    
    Parameters:
    -----------
    method : str
        The method to plot. Options: 'CAGGRIM', 'Sequential', 'Independent', 'Random'
    save_prefix : str
        Prefix for the saved filename
    filepath : str
        Path to the CSV file containing the data
    save_dir : str
        Directory to save the plot in
    """
    # Read data from CSV file
    data = pd.read_csv(filepath)

    # Map method name to corresponding column names
    method_columns = {
        'CAGGRIM': 'CAGGRIM_Score',
        'Sequential': 'Sequential_Score',
        'Independent': 'Independent_Score',
        'Random': 'Random_Score'
    }
    
    # Get the appropriate column name for the selected method
    score_column = method_columns.get(method, 'CAGGRIM_Score')
    
    # Extract unique alpha and beta values
    alphas = sorted(data['Alpha'].unique())
    betas = sorted(data['Beta'].unique())

    # Create a grid for plotting
    beta_mesh, alpha_mesh = np.meshgrid(betas, alphas)

    # Initialize the score grid with NaN values
    score_grid = np.full(alpha_mesh.shape, np.nan)

    # Fill in the score grid with method score values
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            row = data[(data['Alpha'] == alpha) & (data['Beta'] == beta)]
            if not row.empty:
                score_grid[i, j] = row[score_column].values[0]

    # Create 3D figure with specific size
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define reversed color maps (lighter=higher)
    method_colormaps = {
        'CAGGRIM': plt.cm.Reds,     # Red gradient (switched)
        'Sequential': plt.cm.Greens, # Green gradient (switched)
        'Independent': plt.cm.autumn,
        'Random': plt.cm.Blues
    }

    # Calculate method-specific min/max for better color distribution
    valid_scores = score_grid[~np.isnan(score_grid)]
    if len(valid_scores) > 0:
        # Adjust minimum values to avoid too light colors, more aggressively for Sequential and Random
        if method == 'CAGGRIM':
            method_min = max(0, np.min(valid_scores) * 0.8)
        else:
            # For Sequential and Random, use a higher minimum to make light areas darker
            method_min = max(0, np.min(valid_scores) * 0.2)  # More aggressive scaling (was 0.5)
        method_max = np.max(valid_scores)
    else:
        method_min = 150
        method_max = 200

    # Plot the surface with a specific colormap
    surf = ax.plot_surface(beta_mesh, alpha_mesh, score_grid, 
                           cmap=method_colormaps.get(method, plt.cm.viridis),
                           vmin=method_min, vmax=method_max,  # Method-specific range
                           linewidth=0,
                           antialiased=True,
                           edgecolor='none')

    # Set axis labels with large font size (no LaTeX)
    ax.set_xlabel('Beta', fontsize=22, labelpad=15)
    ax.set_ylabel('Alpha', fontsize=22, labelpad=25)  # Increased labelpad to move it higher
    
    # Remove default z-label
    # ax.set_zlabel('AggRI', fontsize=24, labelpad=20)
    
    # Manually place z-label at the top of z-axis
    ax.text(10, 1.0, 200, 'AggRI', fontsize=28, ha='center', va='bottom')
    
    # Adjust z-label position for better visibility
    # ax.zaxis._axinfo['label']['space_factor'] = 3.0

    # # Set title with large font size
    # plt.title(f'{method} Performance Across Alpha and Beta Values', fontsize=24, pad=20)

    # Adjust viewing angle to match reference image
    ax.view_init(elev=25, azim=-45)

    # Flip the axis direction to match the reference image
    ax.invert_xaxis()

    # Add grid lines to make it easier to read values
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)

    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 1.0)
    ax.set_zlim(0, 200)

    # Customize tick labels to be larger
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)

    # Add color bar with custom position and size
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('AggRI Score', fontsize=20, labelpad=15)

    # Ensure the plot fills the figure
    plt.tight_layout(pad=2.5)  # Increase padding to make room for labels

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure with high resolution
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'{save_prefix}_{method.lower()}_heatmap_{timestamp}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)  # Increased padding
    print(f"Plot saved as {save_path}")

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot method performance across alpha and beta values')
    parser.add_argument('--method', type=str, default='CAGGRIM', 
                        choices=['CAGGRIM', 'Sequential', 'Independent', 'Random'],
                        help='Method to plot (default: CAGGRIM)')
    parser.add_argument('--prefix', type=str, default='method_compare',
                        help='Prefix for saved filename (default: alpha_beta)')
    parser.add_argument('--multi', action='store_true',
                        help='Plot multiple methods together (ignores --method)')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['CAGGRIM', 'Sequential', 'Random'],
                        help='Methods to plot when using --multi')
    parser.add_argument('--filepath', type=str, 
                        default='lab6/results/lab6exp1-20250514-224014__agg__borda__custom-init/alpha_beta_sensitivity_results.csv',
                        help='Path to the CSV file containing the data')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save plots in (default: current directory)')
    
    args = parser.parse_args()
    
    if args.multi:
        # Call the multiple methods plotting function
        plot_multiple_methods(methods=args.methods, save_prefix=args.prefix, 
                             filepath=args.filepath, save_dir=args.save_dir)
    else:
        # Call the single method plotting function
        plot_method_performance(method=args.method, save_prefix=args.prefix, 
                               filepath=args.filepath, save_dir=args.save_dir) 