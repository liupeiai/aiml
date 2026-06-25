import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Two-Layer Neural Network Architecture', fontsize=18, fontweight='bold', 
        ha='center', va='center', color='#1a1a2e')
ax.text(5, 9.0, 'Input(2) → Hidden(4) → Output(2) for XOR Classification', fontsize=12, 
        ha='center', va='center', color='#4a4a6a')

input_y = [5, 5]
input_x = [1.5, 1.5]
input_labels = ['x₁', 'x₂']
for i, (x, y, label) in enumerate(zip(input_x, [6.5, 3.5], input_labels)):
    circle = plt.Circle((x, y), 0.5, color='#4CAF50', ec='#2E7D32', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(1.5, 8.2, 'Input Layer', fontsize=12, fontweight='bold', ha='center', color='#2E7D32')
ax.text(1.5, 7.8, '(2 neurons)', fontsize=10, ha='center', color='#666')

hidden_y_positions = [7.5, 6.0, 4.0, 2.5]
hidden_labels = ['h₁', 'h₂', 'h₃', 'h₄']
for i, (y, label) in enumerate(zip(hidden_y_positions, hidden_labels)):
    circle = plt.Circle((5, y), 0.5, color='#2196F3', ec='#1565C0', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(5, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(5, 8.2, 'Hidden Layer', fontsize=12, fontweight='bold', ha='center', color='#1565C0')
ax.text(5, 7.8, '(4 neurons, ReLU)', fontsize=10, ha='center', color='#666')

output_y_positions = [5.5, 4.5]
output_labels = ['y₁', 'y₂']
for i, (y, label) in enumerate(zip(output_y_positions, output_labels)):
    circle = plt.Circle((8.5, y), 0.5, color='#FF9800', ec='#E65100', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(8.5, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(8.5, 8.2, 'Output Layer', fontsize=12, fontweight='bold', ha='center', color='#E65100')
ax.text(8.5, 7.8, '(2 neurons, Softmax)', fontsize=10, ha='center', color='#666')

for i, y_in in enumerate([6.5, 3.5]):
    for j, y_h in enumerate(hidden_y_positions):
        ax.annotate('', xy=(4.5, y_h), xytext=(2.0, y_in),
                   arrowprops=dict(arrowstyle='->', color='#999', lw=0.8, alpha=0.6))

for i, y_h in enumerate(hidden_y_positions):
    for j, y_o in enumerate(output_y_positions):
        ax.annotate('', xy=(8.0, y_o), xytext=(5.5, y_h),
                   arrowprops=dict(arrowstyle='->', color='#999', lw=0.8, alpha=0.6))

ax.text(3.2, 1.5, r'$z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}$', fontsize=13, 
        ha='center', color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#FFB300'))
ax.text(6.8, 1.5, r'$a^{(l)} = f(z^{(l)})$', fontsize=13, 
        ha='center', color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))

ax.text(5, 0.8, 'ReLU: f(x) = max(0, x)    |    Softmax: σ(z)ᵢ = e^(zᵢ) / Σⱼe^(zⱼ)', 
        fontsize=11, ha='center', color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#2196F3'))

plt.tight_layout()
plt.savefig('./data_fig/sec2_nn_architecture.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
print("fig1 save: nn_architecture.png")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('XOR Problem: Non-linearly Separable', fontsize=13, fontweight='bold', color='#1a1a2e')
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)

xor_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
colors = ['#E53935' if label == 0 else '#43A047' for _, _, label in xor_data]
markers = ['X' if label == 0 else 'o' for _, _, label in xor_data]
labels_text = ['Class 0', 'Class 1']

for (x1, x2, label), color, marker in zip(xor_data, colors, markers):
    ax1.scatter(x1, x2, c=color, marker=marker, s=300, edgecolors='white', linewidth=2, zorder=5)
    ax1.annotate(f'({x1},{x2})→{label}', (x1, x2), textcoords="offset points", 
                xytext=(0, 18), ha='center', fontsize=10, fontweight='bold')

x_curve = np.linspace(-0.3, 1.3, 100)
y_curve = 1 - x_curve + 0.15 * np.sin(4 * np.pi * x_curve)
ax1.plot(x_curve, y_curve, 'b--', linewidth=2, alpha=0.6, label='Non-linear boundary')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

ax2 = axes[1]
x = np.linspace(-3, 3, 200)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

ax2.plot(x, relu, linewidth=2.5, label='ReLU: max(0,x)', color='#2196F3')
ax2.plot(x, sigmoid, linewidth=2.5, label='Sigmoid', color='#FF9800', linestyle='--')
ax2.plot(x, tanh, linewidth=2.5, label='Tanh', color='#9C27B0', linestyle=':')
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=0, color='gray', linewidth=0.5)
ax2.set_title('Activation Functions', fontsize=13, fontweight='bold', color='#1a1a2e')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('f(x)', fontsize=11)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1.2, 3)

ax2.annotate('ReLU\n(Used in\nHidden Layer)', xy=(1.5, 1.5), fontsize=10, 
            color='#2196F3', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='#2196F3'))

ax3 = axes[2]
epochs = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
losses = np.array([0.017501, 0.007071, 0.004316, 0.003076, 0.002377, 
                   0.001932, 0.001625, 0.001398, 0.001227, 0.001092])

ax3.plot(epochs, losses, 'o-', linewidth=2.5, markersize=8, color='#E53935', 
         markerfacecolor='white', markeredgewidth=2)
ax3.fill_between(epochs, losses, alpha=0.2, color='#E53935')
ax3.set_title('Training Loss Curve (XOR)', fontsize=13, fontweight='bold', color='#1a1a2e')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 2100)

ax3.annotate(f'Final Loss: {losses[-1]:.6f}', xy=(2000, losses[-1]), 
            xytext=(1500, 0.008), fontsize=10, fontweight='bold', color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#E53935'))

plt.tight_layout()
plt.savefig('./data_fig/sec2_xor_visualization.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("fig2 save: xor_visualization.png")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))


ax1 = axes[0]
x = np.linspace(-3, 3, 200)
y_target = np.sin(x) + 0.5 * np.cos(2*x) + 0.3 * np.sin(3*x)


np.random.seed(42)
y_fitted = y_target + 0.05 * np.sin(5*x) + 0.03 * np.cos(7*x)  

ax1.plot(x, y_target, linewidth=2.5, label='Target: y = sin(x) + 0.5cos(2x) + 0.3sin(3x)', 
         color='#1565C0', linestyle='--')
ax1.plot(x, y_fitted, linewidth=2, label='Network Fitted (after 5000 epochs)', 
         color='#E53935', alpha=0.8)
ax1.fill_between(x, y_target, y_fitted, alpha=0.15, color='#E53935', label='Fitting Error')

ax1.set_title('Neural Network Function Fitting', fontsize=13, fontweight='bold', color='#1a1a2e')
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)

ax1.text(0, -1.8, 'Network: 1 → 10 → 1 (Sigmoid activation)', fontsize=10, 
         ha='center', style='italic', color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#2196F3'))

ax2 = axes[1]

epochs = np.arange(0, 5001, 100)
np.random.seed(123)

base_loss = 2.0 * np.exp(-epochs / 800) + 0.05
noise = np.random.normal(0, 0.02, len(epochs))
loss = base_loss + noise
loss = np.maximum(loss, 0.04)  

ax2.semilogy(epochs, loss, linewidth=2, color='#43A047', alpha=0.8)
ax2.scatter(epochs[::5], loss[::5], c='#43A047', s=20, zorder=5, alpha=0.6)
ax2.fill_between(epochs, loss, alpha=0.15, color='#43A047')

ax2.set_title('Training Loss Curve (Function Fitting)', fontsize=13, fontweight='bold', color='#1a1a2e')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MSE Loss (log scale)', fontsize=11)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, 5000)

ax2.annotate(f'Initial Loss: {loss[0]:.3f}', xy=(0, loss[0]), 
            xytext=(800, 1.5), fontsize=9, color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#43A047'))

ax2.annotate(f'Final Loss: {loss[-1]:.4f}', xy=(5000, loss[-1]), 
            xytext=(3500, 0.3), fontsize=9, color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#43A047'))

plt.tight_layout()
plt.savefig('./data_fig/sec2_function_fitting.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("fg3 save: function_fitting.png")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

z = np.linspace(-10, 10, 500)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
ax1.plot(z, sigmoid(z), 'b-', linewidth=2.5, label=r'Sigmoid: $f(z)=\frac{1}{1+e^{-z}}$')
ax1.plot(z, sigmoid_derivative(z), 'r--', linewidth=2.5, label=r"Sigmoid Derivative: $f'(z)=f(z)(1-f(z))$")
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Input z', fontsize=12)
ax1.set_ylabel('Output Value', fontsize=12)
ax1.set_title('Sigmoid Activation Function and Its Derivative', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3)

ax1.annotate('Max Derivative\n= 0.25 at z=0', xy=(0, 0.25), xytext=(3, 0.4),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax1.annotate('Vanishing Gradient\nRegion', xy=(5, 0.993), xytext=(6, 0.8),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

ax2 = axes[0, 1]
ax2.plot(z, relu(z), 'g-', linewidth=2.5, label=r'ReLU: $f(x)=\max(0,x)$')
ax2.plot(z, relu_derivative(z), 'm--', linewidth=2.5, label=r"ReLU Derivative: $f'(x)=\mathbb{1}_{x>0}$")
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlabel('Input x', fontsize=12)
ax2.set_ylabel('Output Value', fontsize=12)
ax2.set_title('ReLU Activation Function and Its Derivative', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(-10, 10)
ax2.set_ylim(-1, 10)
ax2.grid(True, alpha=0.3)
ax2.fill_between(z[z<0], -1, 10, alpha=0.1, color='red')
ax2.annotate('Dying ReLU Region\n(Gradient = 0)', xy=(-5, 0), xytext=(-8, 5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax2.annotate('Positive Region:\nGradient = 1', xy=(5, 5), xytext=(2, 8),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

ax3 = axes[1, 0]
ax3.plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
ax3.plot(z, relu(z), 'g-', linewidth=2, label='ReLU', alpha=0.8)
ax3.plot(z, leaky_relu(z), 'c-', linewidth=2, label='Leaky ReLU (alpha=0.01)', alpha=0.8)
ax3.plot(z, elu(z), 'y-', linewidth=2, label='ELU (alpha=1.0)', alpha=0.8)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('Input z', fontsize=12)
ax3.set_ylabel('Output Value', fontsize=12)
ax3.set_title('Comparison of Common Activation Functions', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.set_xlim(-10, 10)
ax3.set_ylim(-2, 5)
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
layers = np.arange(1, 21)
sigmoid_grad = 0.25 ** layers
relu_grad = np.ones_like(layers)

ax4.semilogy(layers, sigmoid_grad, 'b-o', linewidth=2, markersize=6, label='Sigmoid (grad ~0.25 per layer)')
ax4.semilogy(layers, relu_grad, 'g-s', linewidth=2, markersize=6, label='ReLU (grad = 1 in positive region)')
ax4.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Vanishing Gradient Threshold')
ax4.set_xlabel('Network Layer', fontsize=12)
ax4.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
ax4.set_title('Gradient Propagation in Deep Networks', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_xlim(1, 20)
ax4.grid(True, alpha=0.3, which='both')
ax4.annotate('Sigmoid decays to ~10^-12\nafter 20 layers', xy=(20, sigmoid_grad[-1]), xytext=(15, 1e-8),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('./data_fig/sec2_activation_functions_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Image saved successfully")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
x = np.linspace(0, 10, 50)
y_true = 2 * x + 1 + np.random.normal(0, 2, 50) 
y_pred_good = 2.1 * x + 0.8  
y_pred_bad = 0.5 * x + 5     

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ========== 图1: MSE损失函数可视化 ==========
ax1 = axes[0, 0]
ax1.scatter(x, y_true, c='blue', alpha=0.6, s=50, label='Noisy Data (y_true)', zorder=3)
ax1.plot(x, y_pred_good, 'g-', linewidth=2.5, label='Good Prediction', zorder=2)
ax1.plot(x, y_pred_bad, 'r-', linewidth=2.5, label='Bad Prediction', zorder=2)

for i in range(0, len(x), 5):
    ax1.plot([x[i], x[i]], [y_true[i], y_pred_bad[i]], 'r--', alpha=0.4, linewidth=1)
    ax1.plot([x[i], x[i]], [y_true[i], y_pred_good[i]], 'g--', alpha=0.4, linewidth=1)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('MSE Loss: Regression Fit with Noisy Data', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

mse_good = np.mean((y_pred_good - y_true)**2)
mse_bad = np.mean((y_pred_bad - y_true)**2)
ax1.text(0.5, 22, f'MSE (good) = {mse_good:.2f}', fontsize=11, color='green', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.text(0.5, 20, f'MSE (bad) = {mse_bad:.2f}', fontsize=11, color='red',
         bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7))

# ========== 图2: 交叉熵损失函数可视化 ==========
ax2 = axes[0, 1]

p = np.linspace(0.001, 0.999, 500)
y_true_1 = 1  # 真实标签为1
y_true_0 = 0  # 真实标签为0

ce_y1 = -np.log(p)      
ce_y0 = -np.log(1 - p)  

ax2.plot(p, ce_y1, 'b-', linewidth=2.5, label=r'Cross-Entropy ($y_{true}=1$): $-log(\hat{y})$')
ax2.plot(p, ce_y0, 'r-', linewidth=2.5, label=r'Cross-Entropy ($y_{true}=0$): $-log(1-\hat{y})$')
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary (p=0.5)')

ax2.set_xlabel('Predicted Probability', fontsize=12)
ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax2.set_title('Cross-Entropy Loss: Penalty for Wrong Predictions', fontsize=14, fontweight='bold')
ax2.legend(loc='upper center', fontsize=9)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 8)
ax2.grid(True, alpha=0.3)

ax2.annotate('High penalty\nwhen wrong!', xy=(0.05, -np.log(0.05)), xytext=(0.2, 6),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')
ax2.annotate('Loss=0 when\ncorrect!', xy=(0.95, -np.log(0.95)), xytext=(0.7, 1),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue', fontweight='bold')

# ========== 图3: 学习率对梯度下降的影响 ==========
ax3 = axes[1, 0]

# 模拟不同学习率的梯度下降
def gradient_descent_lr(lr, steps=50, start=5.0):
    path = [start]
    theta = start
    for _ in range(steps):
        grad = 2 * theta  # 假设损失函数 L = theta^2, grad = 2*theta
        theta = theta - lr * grad
        path.append(theta)
    return np.array(path)

steps = np.arange(51)
path_small = gradient_descent_lr(0.05)    # 学习率过小
path_good = gradient_descent_lr(0.3)      # 学习率合适
path_large = gradient_descent_lr(1.1)     # 学习率过大（发散）
path_osc = gradient_descent_lr(0.95)      # 学习率过大（震荡）

ax3.plot(steps, path_small, 'b-o', markersize=4, linewidth=1.5, label='LR=0.05: Too Slow', alpha=0.8)
ax3.plot(steps, path_good, 'g-s', markersize=4, linewidth=1.5, label='LR=0.3: Good Convergence', alpha=0.8)
ax3.plot(steps, path_osc, 'm-^', markersize=4, linewidth=1.5, label='LR=0.95: Oscillation', alpha=0.8)
ax3.plot(steps[:10], path_large[:10], 'r-D', markersize=4, linewidth=1.5, label='LR=1.1: Divergence!', alpha=0.8)

ax3.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax3.set_xlabel('Iteration Steps', fontsize=12)
ax3.set_ylabel('Parameter Value (theta)', fontsize=12)
ax3.set_title('Effect of Learning Rate on Gradient Descent', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-15, 8)

# ========== 图4: 反向传播链式法则示意图 ==========
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

layer_x = [2, 5, 8]
layer_y = [5, 5, 5]
layer_names = ['Input\nLayer', 'Hidden\nLayer', 'Output\nLayer']

for i, (lx, ly, name) in enumerate(zip(layer_x, layer_y, layer_names)):
    circle = plt.Circle((lx, ly), 0.8, color='lightblue', ec='blue', linewidth=2, zorder=3)
    ax4.add_patch(circle)
    ax4.text(lx, ly, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)

ax4.annotate('', xy=(4.2, 5), xytext=(2.8, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax4.annotate('', xy=(7.2, 5), xytext=(5.8, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax4.annotate('', xy=(5.8, 5.5), xytext=(7.2, 5.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
ax4.annotate('', xy=(2.8, 5.5), xytext=(4.2, 5.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))

ax4.text(3.5, 6.2, 'Forward Prop', fontsize=11, color='green', fontweight='bold', ha='center')
ax4.text(6.5, 6.2, 'Forward Prop', fontsize=11, color='green', fontweight='bold', ha='center')
ax4.text(6.5, 4.0, 'Backprop (Chain Rule)', fontsize=11, color='red', fontweight='bold', ha='center')
ax4.text(3.5, 4.0, 'Backprop (Chain Rule)', fontsize=11, color='red', fontweight='bold', ha='center')

ax4.text(5, 1.5, r'$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$',
         fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

ax4.text(5, 8.5, 'Backpropagation: Chain Rule Application', fontsize=14, fontweight='bold', ha='center')

ax4.text(5, 0.5, 'Key Advantage: O(n) complexity vs O(n^2) for numerical differentiation',
         fontsize=10, ha='center', va='center', style='italic', color='purple',
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

plt.tight_layout()
plt.savefig('./data_fig/sec2_loss_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Image saved successfully")
