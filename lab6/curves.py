import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from pathlib import Path

ASSET_FIGURE_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"
ASSET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def lp_norm_curve(p, s, upper_right=True):
    """
    Generate points on the curve x^p + y^p = 1 for a given value of p.
    For the upper-right family (upper_right=True): directly use the ℓ^p curve
    For the lower-left family (upper_right=False): reflect across y = -x + 1
    
    Parameters:
    p: the p-norm parameter (p ≥ 1)
    s: position parameter in [0,1] to traverse the curve
    upper_right: whether to generate the upper-right or lower-left family
    
    Returns:
    x, y coordinates
    """
    if p == 1:
        # For p=1, the curve is the straight line y = -x + 1 in the first quadrant
        if upper_right:
            x = s
            y = 1 - x
        else:
            x = 1 - s
            y = s
        return x, y
    
    elif np.isinf(p):
        # For p=∞, the curve is the L-shaped corner {(x,1)}∪{(1,y)}
        if upper_right:
            if s <= 0.5:
                # First segment: horizontal from (0,1) to (1,1)
                x = 2 * s
                y = 1
            else:
                # Second segment: vertical from (1,1) to (1,0)
                x = 1
                y = 1 - 2 * (s - 0.5)
        else:
            if s <= 0.5:
                # First segment: vertical from (0,1) to (0,0)
                x = 0
                y = 1 - 2 * s
            else:
                # Second segment: horizontal from (0,0) to (1,0)
                x = 2 * (s - 0.5)
                y = 0
        return x, y
    
    else:
        # For 1 < p < ∞, generate the ℓ^p curve x^p + y^p = 1 in the first quadrant
        if upper_right:
            # Parameterize the curve using angle in [0, π/2]
            theta = s * np.pi/2
            # Convert to rectangular coordinates
            r = 1 / (np.cos(theta)**p + np.sin(theta)**p)**(1/p)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Reflect to the first quadrant and scale to fit the unit square
            x = x
            y = y
        else:
            # For lower-left curve, get the upper-right curve and reflect across y = -x + 1
            x_upper, y_upper = lp_norm_curve(p, 1-s, True)
            x = 1 - y_upper
            y = 1 - x_upper
    
    return x, y

# Create figure for static visualization
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title("ℓ^p Norm Curves in Unit Square", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

# Draw the unit square
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2, label="Unit Square")

# Draw the diagonal lines
s_vals = np.linspace(0, 1, 100)
ax.plot([0, 1], [0, 1], 'purple', linestyle='--', alpha=0.7, label="y = x")
ax.plot([0, 1], [1, 0], 'g-', linewidth=2, label="y = -x + 1 (p = 1)")

# Generate p values for visualization
# Using a non-linear spacing to better see the transition
# p varies from 1 to large values approaching infinity
p_values = [1, 1.5, 2, 3, 5, 10, 50]
p_labels = ["1", "1.5", "2", "3", "5", "10", "∞"]
blue_cmap = plt.cm.Blues(np.linspace(0.4, 1.0, len(p_values)))
red_cmap = plt.cm.Reds(np.linspace(0.4, 1.0, len(p_values)))

# More detailed sampling for smoother curves
s_detailed = np.linspace(0, 1, 200)

# Plot upper right family (blue)
for i, p in enumerate(p_values):
    x_vals = []
    y_vals = []
    
    # Use p=1000 to approximate p=∞
    actual_p = 1000 if p_labels[i] == "∞" else p
    
    for s in s_detailed:
        x, y = lp_norm_curve(actual_p, s, upper_right=True)
        x_vals.append(x)
        y_vals.append(y)
    
    ax.plot(x_vals, y_vals, color=blue_cmap[i], linewidth=2, 
            label=f"Upper Right (p = {p_labels[i]})")

# Plot lower left family (red)
for i, p in enumerate(p_values):
    x_vals = []
    y_vals = []
    
    # Use p=1000 to approximate p=∞
    actual_p = 1000 if p_labels[i] == "∞" else p
    
    for s in s_detailed:
        x, y = lp_norm_curve(actual_p, s, upper_right=False)
        x_vals.append(x)
        y_vals.append(y)
    
    ax.plot(x_vals, y_vals, color=red_cmap[i], linewidth=2, 
            label=f"Lower Left (p = {p_labels[i]})")

# Add shading to emphasize the two halves of the square
vertices_upper = [(0,1), (1,1), (1,0), (0.5, 0.5)]
vertices_lower = [(0,0), (1,0), (0,1), (0.5, 0.5)]

upper_polygon = plt.Polygon(vertices_upper, color='blue', alpha=0.05)
lower_polygon = plt.Polygon(vertices_lower, color='red', alpha=0.05)
ax.add_patch(upper_polygon)
ax.add_patch(lower_polygon)

# Place legend outside the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(ASSET_FIGURE_DIR / 'lp_norm_curves.png', dpi=300)
plt.show()

# Create animation to show the smooth transition
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title("ℓ^p Norm Curves - Animation", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

# Draw the unit square
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2)

# Draw the diagonal lines
ax.plot([0, 1], [0, 1], 'purple', linestyle='--', alpha=0.7)
ax.plot([0, 1], [1, 0], 'g-', linewidth=2)

# Add shading to emphasize the two halves of the square
ax.add_patch(plt.Polygon(vertices_upper, color='blue', alpha=0.05))
ax.add_patch(plt.Polygon(vertices_lower, color='red', alpha=0.05))

# Initialize lines for animation
blue_line, = ax.plot([], [], 'b-', linewidth=2.5, label="Upper Right Family")
red_line, = ax.plot([], [], 'r-', linewidth=2.5, label="Lower Left Family")
p_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

def init():
    blue_line.set_data([], [])
    red_line.set_data([], [])
    p_text.set_text("")
    return blue_line, red_line, p_text

def animate(frame):
    # Use a non-linear mapping to better visualize the transition
    # This gives more frames to the early p values where change is most visible
    if frame <= 50:
        # p goes from 1 to 5
        t = frame / 50
        p = 1 + 4 * t
    elif frame <= 80:
        # p goes from 5 to 20
        t = (frame - 50) / 30
        p = 5 + 15 * t
    else:
        # p approaches infinity
        t = (frame - 80) / 20
        p = 20 + 980 * t  # approximating infinity with 1000
    
    # For infinity, use a very large p value
    p_display = "∞" if p > 500 else f"{p:.1f}"
    
    # Calculate points for upper right family
    x_upper = []
    y_upper = []
    for s in s_detailed:
        x, y = lp_norm_curve(p, s, upper_right=True)
        x_upper.append(x)
        y_upper.append(y)
    
    # Calculate points for lower left family
    x_lower = []
    y_lower = []
    for s in s_detailed:
        x, y = lp_norm_curve(p, s, upper_right=False)
        x_lower.append(x)
        y_lower.append(y)
    
    blue_line.set_data(x_upper, y_upper)
    red_line.set_data(x_lower, y_lower)
    p_text.set_text(f"p = {p_display}")
    
    return blue_line, red_line, p_text

# Create animation
anim = FuncAnimation(fig, animate, frames=101, init_func=init, blit=True, interval=50)
plt.tight_layout()

# Uncomment to save the animation
anim.save(ASSET_FIGURE_DIR / 'lp_norm_curves_animation.gif', writer='pillow', fps=20, dpi=100)

plt.show()
