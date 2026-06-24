import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# NLINEX components using manuscript notation
# ---------------------------------------------------------
def exp_core(D, k, c):
    return k * (np.exp(c * D) - 1)

def linear_cancel(D, k, c):
    return -k * c * D

def quadratic(D, c):
    return c * (D ** 2)

def nlinex(D, k, c):
    return exp_core(D, k, c) + linear_cancel(D, k, c) + quadratic(D, c)

# ---------------------------------------------------------
# Expanded D range so all terms are visible
# ---------------------------------------------------------
D = np.linspace(-10, 10, 800)

# ---------------------------------------------------------
# Your specified (k, c) pairs
# ---------------------------------------------------------
pairs = [
    (0.5, 3),
    (1, 2),
    (1.5, 1),
    (1.8, 0.5),
    (2.02, 0.206),   # Optimised
    (3, 0.09)
]

# ---------------------------------------------------------
# FULL RESET: start a brand‑new figure (kills all cached labels)
# ---------------------------------------------------------
plt.close('all')
fig, axes = plt.subplots(3, 2, figsize=(18, 22))
axes = axes.flatten()

# ---------------------------------------------------------
# Create clean dummy handles for ONE global legend
# ---------------------------------------------------------
h_exp, = plt.plot([], [], linewidth=2, label="Exponential Core")
h_lin, = plt.plot([], [], linewidth=2, label="Linear Cancellation")
h_quad, = plt.plot([], [], linewidth=2, label="Quadratic Term")
h_total, = plt.plot([], [], linewidth=3, color="black", label="Final NLINEX")

global_handles = [h_exp, h_lin, h_quad, h_total]

# ---------------------------------------------------------
# Plot each subplot WITHOUT labels
# ---------------------------------------------------------
for ax, (k, c) in zip(axes, pairs):

    # Hard reset of axes
    ax.cla()

    L_exp = exp_core(D, k, c)
    L_lin = linear_cancel(D, k, c)
    L_quad = quadratic(D, c)
    L_total = nlinex(D, k, c)

    # Plot WITHOUT labels (prevents le2/le8 forever)
    ax.plot(D, L_exp, linewidth=2)
    ax.plot(D, L_lin, linewidth=2)
    ax.plot(D, L_quad, linewidth=2)
    ax.plot(D, L_total, linewidth=3, color="black")

    title = f"k={k}, c={c}"
    if abs(k - 2.02) < 1e-6 and abs(c - 0.206) < 1e-6:
        title += "  (Optimised)"
    ax.set_title(title, fontsize=16)

    ax.grid(True, alpha=0.3)
    ax.autoscale(enable=True, axis='y')

# ---------------------------------------------------------
# Add ONE global legend at the bottom
# ---------------------------------------------------------
fig.legend(
    handles=global_handles,
    loc="lower center",
    fontsize=14,
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

plt.subplots_adjust(hspace=0.55, wspace=0.25, bottom=0.08)
plt.show()

