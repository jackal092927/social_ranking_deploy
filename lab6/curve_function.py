import numpy as np
import matplotlib.pyplot as plt

def lp_curve(convex=True, p=2, square_size=1, num_points=100):
    """
    Generate a parametrized ℓ^p curve in a square of specified size.
    
    Parameters:
    -----------
    convex : bool
        If True, generates curves that are convex toward the origin (upper-right family)
        If False, generates curves that are concave from the origin (lower-left family)
    p : float
        The p-norm parameter (1 ≤ p ≤ infinity)
        p=1 gives a straight line
        p=2 gives a quarter circle (for convex=True)
        p→∞ approaches the L-shaped corner path
    square_size : float
        The size of the bounding square [0, square_size] × [0, square_size]
    num_points : int
        Number of points to generate along the curve
        
    Returns:
    --------
    x : numpy.ndarray
        x-coordinates of the curve points
    y : numpy.ndarray
        y-coordinates of the curve points
    """
    # Handle p = infinity
    if np.isinf(p):
        if convex:
            # L-shaped path from (0,square_size) to (square_size,0)
            half = num_points // 2
            x1 = np.linspace(0, square_size, half)
            y1 = np.ones_like(x1) * square_size
            
            x2 = np.ones(num_points - half) * square_size
            y2 = np.linspace(square_size, 0, num_points - half)
            
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
        else:
            # L-shaped path from (0,square_size) to (square_size,0)
            half = num_points // 2
            x1 = np.zeros(half)
            y1 = np.linspace(square_size, 0, half)
            
            x2 = np.linspace(0, square_size, num_points - half)
            y2 = np.zeros_like(x2)
            
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
        
        return x, y
    
    # For p = 1, the curve is a straight line
    elif p == 1:
        if convex:
            # Line from (0,square_size) to (square_size,0)
            x = np.linspace(0, square_size, num_points)
            y = square_size - x
        else:
            # Same line but parameterized in reverse direction
            x = np.linspace(square_size, 0, num_points)
            y = square_size - x
        
        return x, y
    
    # For 1 < p < infinity, generate the ℓ^p curve
    else:
        # Parameterize the curve by angle in [0, π/2]
        theta = np.linspace(0, np.pi/2, num_points)
        
        # Calculate radius for the pth power
        r = square_size / (np.cos(theta)**p + np.sin(theta)**p)**(1/p)
        
        # Convert to rectangular coordinates
        x_raw = r * np.cos(theta)
        y_raw = r * np.sin(theta)
        
        if convex:
            # Upper-right family (convex toward origin)
            x = x_raw
            y = y_raw
        else:
            # Lower-left family (concave from origin)
            # Reflect across the line y = -x + square_size
            x = square_size - y_raw
            y = square_size - x_raw
        
        return x, y

# Example usage and visualization
def plot_examples():
    """Generate an example plot showing various curves"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set square size
    size = 10
    
    # Draw the bounding square
    ax.plot([0, size, size, 0, 0], [0, 0, size, size, 0], 'k-', linewidth=2)
    
    # Draw reference lines
    ax.plot([0, size], [0, size], 'purple', linestyle='--', alpha=0.5)
    ax.plot([0, size], [size, 0], 'green', linestyle='--', alpha=0.5)
    
    # Plot several convex curves with different p values
    p_values = [1, 1.5, 2, 3, 5, 10, float('inf')]
    p_labels = ["1", "1.5", "2", "3", "5", "10", "∞"]
    
    for i, p in enumerate(p_values):
        # Convex curves (blue)
        x, y = lp_curve(convex=True, p=p, square_size=size)
        ax.plot(x, y, 'b-', alpha=0.7, linewidth=2, label=f"Convex p={p_labels[i]}" if i == 0 else "")
        
        # Concave curves (red)
        x, y = lp_curve(convex=False, p=p, square_size=size)
        ax.plot(x, y, 'r-', alpha=0.7, linewidth=2, label=f"Concave p={p_labels[i]}" if i == 0 else "")
    
    # Annotate some of the curves
    for i, p in enumerate([1, 2, float('inf')]):
        label = "1" if p == 1 else ("2" if p == 2 else "∞")
        
        # Get points from the convex curve for annotation
        x, y = lp_curve(convex=True, p=p, square_size=size)
        mid_idx = len(x) // 2
        ax.text(x[mid_idx], y[mid_idx], f"p={label}", color='blue', fontsize=12)
        
        # Get points from the concave curve for annotation
        x, y = lp_curve(convex=False, p=p, square_size=size)
        mid_idx = len(x) // 2
        ax.text(x[mid_idx], y[mid_idx], f"p={label}", color='red', fontsize=12)
    
    ax.set_xlim(-1, size+1)
    ax.set_ylim(-1, size+1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"ℓ^p Curves in a {size}×{size} Square", fontsize=16)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_examples()