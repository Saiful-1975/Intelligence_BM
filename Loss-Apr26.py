import numpy as np
import matplotlib.pyplot as plt

# Predicted values from -1 to 1
y_pred = np.linspace(-1, 1, 500)
y_true = 0.50  # Ground truth updated

# Clip predicted values to avoid log(0)
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

# -----------------------------
# 1. MSE
# -----------------------------
mse = (y_pred - y_true) ** 2

# -----------------------------
# 2. MAE
# -----------------------------
mae = np.abs(y_pred - y_true)

# -----------------------------
# 3. BCE
# -----------------------------
bce = - (y_true * np.log(y_pred_clipped) +
         (1 - y_true) * np.log(1 - y_pred_clipped))

# -----------------------------
# 4. NLINEX (Optimised)
# -----------------------------
def nlinex_loss(y_pred, y_true, k, c):
    D = y_pred - y_true
    return k * (np.exp(c * D) + c * D**2 - c * D - 1)

nlinex_opt = nlinex_loss(y_pred, y_true, k=2.020, c=0.206)

# -----------------------------
# 5. Focal Loss
# -----------------------------
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    pt = np.where(y_true == 1, y_pred_clipped, 1 - y_pred_clipped)
    return -alpha * (1 - pt)**gamma * np.log(pt)

focal_loss_values = focal_loss(y_pred, y_true)

# -----------------------------
# 6. Lovász (shape approximation)
# -----------------------------
def lovasz_loss(y_pred, y_true):
    error = y_pred - y_true
    delta_j = np.maximum(0, 1 - np.where(y_pred >= 0.5, 1, 0) * y_true)
    return delta_j * np.abs(error)

lovasz_loss_values = lovasz_loss(y_pred, y_true)

# -----------------------------
# 7. Dice Loss (smooth)
# -----------------------------
def dice_loss(y_pred, y_true, eps=1e-7):
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    intersection = y_pred_clipped * y_true
    return 1 - (2 * intersection + eps) / (y_pred_clipped + y_true + eps)

dice_loss_values = dice_loss(y_pred, y_true)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(10, 8))

plt.plot(y_pred, mse, label='MSE', color='blue')
plt.plot(y_pred, mae, label='MAE', color='orange')
plt.plot(y_pred, bce, label='BCE', color='green')

plt.plot(y_pred, nlinex_opt, label='Optimised NLINEX (k=2.020, c=0.206)',
         color='black', linestyle='--', linewidth=2)

plt.plot(y_pred, focal_loss_values, label='Focal Loss', color='purple')
plt.plot(y_pred, lovasz_loss_values, label='Lovász Loss', color='magenta')
plt.plot(y_pred, dice_loss_values, label='Dice Loss', color='brown')

plt.xlabel('Predicted Value (-1 to 1)')
plt.ylabel('Loss')
plt.title('Comparative Analysis of Loss Function Shapes (Ground Truth = 0.50)')

# ⭐ Legend moved 1 cm left
plt.legend(loc='upper right', bbox_to_anchor=(0.96, 1))

plt.grid(True)
plt.ylim(0, 6)

# -----------------------------
# Caption BELOW the plot
# -----------------------------
plt.figtext(
    0.5, -0.08,
    "Figure 1: Comparative analysis of loss function shapes for predicted values ranging from "
    "-1 to 1, with a fixed ground truth of 0.25",
    wrap=True, ha='center', fontsize=11
)

plt.tight_layout()
plt.savefig('loss_functions_7shape.png', dpi=600, bbox_inches='tight')
plt.show()
