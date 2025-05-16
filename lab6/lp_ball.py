import numpy as np
import matplotlib.pyplot as plt

def lp_ball(p, num=300, lim=2.0):
    """
    Return a grid of points (X, Y) that lie in the set {|x|^p + |y|^p <= 1}
    for the real number p.
    """
    # Make a grid from -lim to lim in both directions
    xs = np.linspace(-lim, lim, num)
    ys = np.linspace(-lim, lim, num)
    X, Y = np.meshgrid(xs, ys)
    
    # For negative p, to avoid issues at x=0.0, we handle zero carefully.
    # We'll do an element-wise safe evaluation:
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute |X|^p + |Y|^p
        # If p < 0 and X=0.0 => |X|^p => 0^p => +inf, so we must be cautious
        vals = np.power(np.abs(X), p) + np.power(np.abs(Y), p)
    
    # The mask is True where the p-"sum" is <= 1
    mask = (vals <= 1.0)
    return X[mask], Y[mask]

# Example: visualize a few distinct p values
for p in [0, 0.1, 0.25, 0.5, 1, 2, 4,8, 1000 ]:  # np.inf is a symbolic "infinity"
    X_in, Y_in = lp_ball(p)
    plt.figure()
    plt.scatter(X_in, Y_in, s=1)  # no special color
    plt.title(f"Points satisfying |x|^{p} + |y|^{p} <= 1,  p = {p}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-2,2]); plt.ylim([-2,2])
    plt.show()
