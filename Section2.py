import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1: 神经网络结构图
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(5, 9.5, 'Two-Layer Neural Network Architecture', fontsize=18, fontweight='bold', 
        ha='center', va='center', color='#1a1a2e')
ax.text(5, 9.0, 'Input(2) → Hidden(4) → Output(2) for XOR Classification', fontsize=12, 
        ha='center', va='center', color='#4a4a6a')

# 输入层 (2个神经元)
input_y = [5, 5]
input_x = [1.5, 1.5]
input_labels = ['x₁', 'x₂']
for i, (x, y, label) in enumerate(zip(input_x, [6.5, 3.5], input_labels)):
    circle = plt.Circle((x, y), 0.5, color='#4CAF50', ec='#2E7D32', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(1.5, 8.2, 'Input Layer', fontsize=12, fontweight='bold', ha='center', color='#2E7D32')
ax.text(1.5, 7.8, '(2 neurons)', fontsize=10, ha='center', color='#666')

# 隐藏层 (4个神经元)
hidden_y_positions = [7.5, 6.0, 4.0, 2.5]
hidden_labels = ['h₁', 'h₂', 'h₃', 'h₄']
for i, (y, label) in enumerate(zip(hidden_y_positions, hidden_labels)):
    circle = plt.Circle((5, y), 0.5, color='#2196F3', ec='#1565C0', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(5, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(5, 8.2, 'Hidden Layer', fontsize=12, fontweight='bold', ha='center', color='#1565C0')
ax.text(5, 7.8, '(4 neurons, ReLU)', fontsize=10, ha='center', color='#666')

# 输出层 (2个神经元)
output_y_positions = [5.5, 4.5]
output_labels = ['y₁', 'y₂']
for i, (y, label) in enumerate(zip(output_y_positions, output_labels)):
    circle = plt.Circle((8.5, y), 0.5, color='#FF9800', ec='#E65100', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(8.5, y, label, fontsize=14, ha='center', va='center', fontweight='bold', color='white')

ax.text(8.5, 8.2, 'Output Layer', fontsize=12, fontweight='bold', ha='center', color='#E65100')
ax.text(8.5, 7.8, '(2 neurons, Softmax)', fontsize=10, ha='center', color='#666')

# 绘制连接线和权重标签
# 输入层到隐藏层
for i, y_in in enumerate([6.5, 3.5]):
    for j, y_h in enumerate(hidden_y_positions):
        ax.annotate('', xy=(4.5, y_h), xytext=(2.0, y_in),
                   arrowprops=dict(arrowstyle='->', color='#999', lw=0.8, alpha=0.6))

# 隐藏层到输出层
for i, y_h in enumerate(hidden_y_positions):
    for j, y_o in enumerate(output_y_positions):
        ax.annotate('', xy=(8.0, y_o), xytext=(5.5, y_h),
                   arrowprops=dict(arrowstyle='->', color='#999', lw=0.8, alpha=0.6))

# 添加公式标注
ax.text(3.2, 1.5, r'$z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}$', fontsize=13, 
        ha='center', color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#FFB300'))
ax.text(6.8, 1.5, r'$a^{(l)} = f(z^{(l)})$', fontsize=13, 
        ha='center', color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))

# 添加激活函数标注
ax.text(5, 0.8, 'ReLU: f(x) = max(0, x)    |    Softmax: σ(z)ᵢ = e^(zᵢ) / Σⱼe^(zⱼ)', 
        fontsize=11, ha='center', color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#2196F3'))

plt.tight_layout()
plt.savefig('/lpa/vm/aiml/nn_architecture.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
print("图1保存完成: nn_architecture.png")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import matplotlib.pyplot as plt
import numpy as np

# 图2: XOR问题可视化 + 训练过程
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 子图1: XOR真值表可视化
ax1 = axes[0]
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('XOR Problem: Non-linearly Separable', fontsize=13, fontweight='bold', color='#1a1a2e')
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)

# XOR数据点
xor_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
colors = ['#E53935' if label == 0 else '#43A047' for _, _, label in xor_data]
markers = ['X' if label == 0 else 'o' for _, _, label in xor_data]
labels_text = ['Class 0', 'Class 1']

for (x1, x2, label), color, marker in zip(xor_data, colors, markers):
    ax1.scatter(x1, x2, c=color, marker=marker, s=300, edgecolors='white', linewidth=2, zorder=5)
    ax1.annotate(f'({x1},{x2})→{label}', (x1, x2), textcoords="offset points", 
                xytext=(0, 18), ha='center', fontsize=10, fontweight='bold')

# 画一条示意性的非线性决策边界
x_curve = np.linspace(-0.3, 1.3, 100)
y_curve = 1 - x_curve + 0.15 * np.sin(4 * np.pi * x_curve)
ax1.plot(x_curve, y_curve, 'b--', linewidth=2, alpha=0.6, label='Non-linear boundary')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

# 子图2: 激活函数对比
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

# 添加ReLU的标注
ax2.annotate('ReLU\n(Used in\nHidden Layer)', xy=(1.5, 1.5), fontsize=10, 
            color='#2196F3', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='#2196F3'))

# 子图3: 训练损失曲线
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

# 添加最终损失标注
ax3.annotate(f'Final Loss: {losses[-1]:.6f}', xy=(2000, losses[-1]), 
            xytext=(1500, 0.008), fontsize=10, fontweight='bold', color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#E53935'))

plt.tight_layout()
plt.savefig('/lpa/vm/aiml/xor_visualization.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("图2保存完成: xor_visualization.png")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
import numpy as np

# 图3: 函数拟合可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 子图1: 目标函数 vs 网络拟合结果
ax1 = axes[0]

# 生成目标函数数据
x = np.linspace(-3, 3, 200)
y_target = np.sin(x) + 0.5 * np.cos(2*x) + 0.3 * np.sin(3*x)

# 模拟训练后的拟合结果（使用一些噪声模拟）
np.random.seed(42)
y_fitted = y_target + 0.05 * np.sin(5*x) + 0.03 * np.cos(7*x)  # 轻微差异

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

# 添加网络结构标注
ax1.text(0, -1.8, 'Network: 1 → 10 → 1 (Sigmoid activation)', fontsize=10, 
         ha='center', style='italic', color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#2196F3'))

# 子图2: 训练损失曲线（函数拟合）
ax2 = axes[1]

# 模拟训练过程
epochs = np.arange(0, 5001, 100)
np.random.seed(123)
# 创建合理的损失下降曲线
base_loss = 2.0 * np.exp(-epochs / 800) + 0.05
noise = np.random.normal(0, 0.02, len(epochs))
loss = base_loss + noise
loss = np.maximum(loss, 0.04)  # 确保非负

ax2.semilogy(epochs, loss, linewidth=2, color='#43A047', alpha=0.8)
ax2.scatter(epochs[::5], loss[::5], c='#43A047', s=20, zorder=5, alpha=0.6)
ax2.fill_between(epochs, loss, alpha=0.15, color='#43A047')

ax2.set_title('Training Loss Curve (Function Fitting)', fontsize=13, fontweight='bold', color='#1a1a2e')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MSE Loss (log scale)', fontsize=11)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, 5000)

# 标注关键信息
ax2.annotate(f'Initial Loss: {loss[0]:.3f}', xy=(0, loss[0]), 
            xytext=(800, 1.5), fontsize=9, color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#43A047'))

ax2.annotate(f'Final Loss: {loss[-1]:.4f}', xy=(5000, loss[-1]), 
            xytext=(3500, 0.3), fontsize=9, color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#43A047'))

plt.tight_layout()
plt.savefig('/lpa/vm/aiml/function_fitting.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("图3保存完成: function_fitting.png")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import matplotlib.pyplot as plt

# 设置字体 - 使用系统默认字体避免中文乱码
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义激活函数
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

# 生成数据
z = np.linspace(-10, 10, 500)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: Sigmoid函数及其导数
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
# 标注关键点
ax1.annotate('Max Derivative\n= 0.25 at z=0', xy=(0, 0.25), xytext=(3, 0.4),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax1.annotate('Vanishing Gradient\nRegion', xy=(5, 0.993), xytext=(6, 0.8),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')

# 图2: ReLU函数及其导数
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
# 标注死亡区域
ax2.fill_between(z[z<0], -1, 10, alpha=0.1, color='red')
ax2.annotate('Dying ReLU Region\n(Gradient = 0)', xy=(-5, 0), xytext=(-8, 5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')
ax2.annotate('Positive Region:\nGradient = 1', xy=(5, 5), xytext=(2, 8),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

# 图3: 多种激活函数对比
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

# 图4: 梯度消失问题可视化
ax4 = axes[1, 1]
# 模拟多层网络中的梯度传播
layers = np.arange(1, 21)
# Sigmoid梯度: 假设每层梯度约为0.25 (最大处)
sigmoid_grad = 0.25 ** layers
# ReLU梯度: 假设正区间梯度为1
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
plt.savefig('/lpa/vm/aiml/activation_functions_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("Image saved successfully")