import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 确保输出目录存在
import os
os.makedirs('./data_fig/', exist_ok=True)

# ============================================
# 图1: XOR问题数据分布与线性不可分性
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图: XOR数据点分布
ax1 = axes[0]
XOR_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_labels = [0, 1, 1, 0]
colors = ['#E74C3C' if l == 1 else '#3498DB' for l in XOR_labels]

for i, (x, y) in enumerate(XOR_data):
    ax1.scatter(x, y, c=colors[i], s=400, zorder=5, edgecolors='black', linewidths=2)
    ax1.annotate(f'({x},{y})\n→ {XOR_labels[i]}', (x, y), 
                textcoords="offset points", xytext=(0, 18), 
                ha='center', fontsize=11, fontweight='bold')

# 绘制线性分类尝试（虚线）
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='线性分类边界尝试')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('输入 X₁', fontsize=12)
ax1.set_ylabel('输入 X₂', fontsize=12)
ax1.set_title('(a) XOR问题数据分布', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)

# 添加图例说明
red_patch = mpatches.Patch(color='#E74C3C', label='输出=1')
blue_patch = mpatches.Patch(color='#3498DB', label='输出=0')
ax1.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=10)

# 右图: 线性不可分示意
ax2 = axes[1]
# 绘制决策区域
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
# 模拟线性分类器的决策边界
Z_linear = (xx + yy > 1).astype(int)
ax2.contourf(xx, yy, Z_linear, levels=[-0.5, 0.5, 1.5], colors=['#3498DB', '#E74C3C'], alpha=0.3)

for i, (x, y) in enumerate(XOR_data):
    ax2.scatter(x, y, c=colors[i], s=400, zorder=5, edgecolors='black', linewidths=2)
    
ax2.plot([0, 1], [1, 0], 'k--', linewidth=2, label='理想非线性边界')
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel('输入 X₁', fontsize=12)
ax2.set_ylabel('输入 X₂', fontsize=12)
ax2.set_title('(b) 线性分类器无法正确分类', fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('./data_fig/sec3_xor_problem.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图1已保存: XOR问题数据分布与线性不可分性")


#%%%%%


# ============================================
# 图2: 两层神经网络结构示意图
# ============================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# 绘制层
layer_colors = {'input': '#E8F4FD', 'hidden': '#D5F5E3', 'output': '#FADBD8'}
layer_border = {'input': '#3498DB', 'hidden': '#27AE60', 'output': '#E74C3C'}

def draw_layer(ax, x, y_start, n_neurons, layer_name, color, border_color, neuron_labels=None):
    neuron_height = 0.6
    gap = 0.3
    total_height = n_neurons * neuron_height + (n_neurons - 1) * gap
    y_center = y_start - total_height / 2
    
    # 绘制层背景框
    rect = FancyBboxPatch((x - 0.8, y_center - 0.3), 1.6, total_height + 0.6,
                          boxstyle="round,pad=0.1", facecolor=color, edgecolor=border_color, 
                          linewidth=2, alpha=0.5)
    ax.add_patch(rect)
    
    # 绘制神经元
    for i in range(n_neurons):
        y = y_center + (n_neurons - 1 - i) * (neuron_height + gap) + neuron_height / 2
        circle = Circle((x, y), 0.25, facecolor='white', edgecolor=border_color, linewidth=2, zorder=5)
        ax.add_patch(circle)
        if neuron_labels and i < len(neuron_labels):
            ax.text(x, y, neuron_labels[i], ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)
    
    # 层标签
    ax.text(x, y_center - 0.8, layer_name, ha='center', va='top', fontsize=12, fontweight='bold', color=border_color)
    return y_center, total_height

# 输入层
y_in, h_in = draw_layer(ax, 1.5, 3.5, 2, '输入层\n(Input)', layer_colors['input'], layer_border['input'], ['x₁', 'x₂'])

# 隐藏层
y_hid, h_hid = draw_layer(ax, 5, 3.5, 4, '隐藏层\n(Hidden)', layer_colors['hidden'], layer_border['hidden'], ['h₁', 'h₂', 'h₃', 'h₄'])

# 输出层
y_out, h_out = draw_layer(ax, 8.5, 3.5, 2, '输出层\n(Output)', layer_colors['output'], layer_border['output'], ['ŷ₁', 'ŷ₂'])

# 绘制连接线和权重标注
# 输入到隐藏层
np.random.seed(42)
for i in range(2):
    y_i = y_in + (2 - 1 - i) * (0.6 + 0.3) + 0.3
    for j in range(4):
        y_j = y_hid + (4 - 1 - j) * (0.6 + 0.3) + 0.3
        ax.annotate('', xy=(5 - 0.25, y_j), xytext=(1.5 + 0.25, y_i),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))

# 隐藏到输出层
for i in range(4):
    y_i = y_hid + (4 - 1 - i) * (0.6 + 0.3) + 0.3
    for j in range(2):
        y_j = y_out + (2 - 1 - j) * (0.6 + 0.3) + 0.3
        ax.annotate('', xy=(8.5 - 0.25, y_j), xytext=(5 + 0.25, y_i),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6))

# 添加激活函数标注
ax.text(5, 5.5, 'ReLU', ha='center', va='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='#F9E79F', edgecolor='#F39C12', linewidth=1.5))
ax.text(8.5, 5.5, 'Softmax', ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#F9E79F', edgecolor='#F39C12', linewidth=1.5))

# 添加数学公式
ax.text(3.2, 3.5, r'$W^{(1)}, b^{(1)}$', ha='center', va='center', fontsize=11, 
        color='#27AE60', fontweight='bold')
ax.text(6.8, 3.5, r'$W^{(2)}, b^{(2)}$', ha='center', va='center', fontsize=11,
        color='#E74C3C', fontweight='bold')

# 标题
ax.text(5, 6.5, '图2  两层神经网络结构示意图', ha='center', va='center', fontsize=14, fontweight='bold')

# 添加维度标注
ax.text(1.5, 0.5, '2维输入', ha='center', fontsize=10, color='#3498DB')
ax.text(5, 0.5, '4个隐藏神经元', ha='center', fontsize=10, color='#27AE60')
ax.text(8.5, 0.5, '2维输出', ha='center', fontsize=10, color='#E74C3C')

plt.tight_layout()
plt.savefig('./data_fig/sec3_network_architecture.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图2已保存: 两层神经网络结构示意图")


#%%%%%%%%%%%%%%%


# ============================================
# 图3: XOR训练过程——损失曲线与决策边界演化
# ============================================
fig = plt.figure(figsize=(16, 10))

# 模拟训练过程
np.random.seed(42)
epochs = 2000
loss_history = []
# 模拟损失下降曲线
for e in range(epochs):
    if e < 100:
        loss = 0.7 * np.exp(-e/50) + 0.3
    elif e < 500:
        loss = 0.3 * np.exp(-(e-100)/200) + 0.05
    elif e < 1500:
        loss = 0.05 * np.exp(-(e-500)/500) + 0.001
    else:
        loss = 0.001 + 0.0005 * np.random.randn()
    loss_history.append(max(loss, 0.0005))

# 上部分：损失曲线
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(epochs), loss_history, color='#2980B9', linewidth=1.2, alpha=0.8)
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax1.set_ylabel('交叉熵损失', fontsize=12)
ax1.set_title('(a) 训练损失曲线', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.001, color='red', linestyle='--', alpha=0.5, label='目标损失≈0.001')
ax1.legend(fontsize=10)

# 标注关键阶段
ax1.annotate('快速下降期', xy=(200, 0.15), xytext=(400, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax1.annotate('缓慢收敛期', xy=(1200, 0.003), xytext=(1400, 0.02),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

# 上部分右：决策边界演化
ax2 = plt.subplot(2, 2, 2)
# 模拟不同epoch的决策边界
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))

# 训练后期的决策边界（非线性）
def decision_boundary(x1, x2, epoch_ratio):
    # 模拟网络学习到的非线性边界
    w1 = 5 * epoch_ratio
    w2 = 5 * epoch_ratio
    b = -2.5 * epoch_ratio
    return 1 / (1 + np.exp(-(w1 * x1 + w2 * x2 + b + 3 * epoch_ratio * x1 * x2)))

Z = decision_boundary(xx, yy, 1.0)
contour = ax2.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

XOR_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_labels = [0, 1, 1, 0]
colors = ['#E74C3C' if l == 1 else '#3498DB' for l in XOR_labels]
for i, (x, y) in enumerate(XOR_data):
    ax2.scatter(x, y, c=colors[i], s=300, zorder=5, edgecolors='black', linewidths=2)

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel('输入 X₁', fontsize=12)
ax2.set_ylabel('输入 X₂', fontsize=12)
ax2.set_title('(b) 学习到的非线性决策边界', fontsize=13, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
plt.colorbar(contour, ax=ax2, shrink=0.8)

# 下部分：训练不同阶段的预测概率
ax3 = plt.subplot(2, 1, 2)
snapshots = [0, 100, 500, 1000, 2000]
snapshot_labels = ['Epoch 0\n(初始)', 'Epoch 100\n(快速学习)', 'Epoch 500\n(中期)', 'Epoch 1000\n(后期)', 'Epoch 2000\n(收敛)']

x_pos = np.arange(4)
width = 0.15
inputs = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
targets = [0, 1, 1, 0]

for idx, (epoch_snap, label) in enumerate(zip(snapshots, snapshot_labels)):
    ratio = min(epoch_snap / 2000, 1.0)
    # 模拟预测概率
    if epoch_snap == 0:
        probs = [0.5, 0.5, 0.5, 0.5]
    else:
        probs = [
            max(0.5 - ratio * 0.49, 0.01),  # (0,0) -> 0
            min(0.5 + ratio * 0.49, 0.99),  # (0,1) -> 1
            min(0.5 + ratio * 0.49, 0.99),  # (1,0) -> 1
            max(0.5 - ratio * 0.49, 0.01),  # (1,1) -> 0
        ]
    bars = ax3.bar(x_pos + idx * width - 2*width, probs, width, label=label, alpha=0.85)

ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel('输入样本', fontsize=12)
ax3.set_ylabel('预测概率 (类别1)', fontsize=12)
ax3.set_title('(c) 不同训练阶段的预测概率变化', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(inputs)
ax3.set_ylim(0, 1.1)
ax3.legend(loc='upper right', fontsize=9, ncol=3)
ax3.grid(True, alpha=0.3, axis='y')

# 添加目标标记
for i, t in enumerate(targets):
    ax3.annotate(f'目标:{t}', (i, 1.05), ha='center', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('./data_fig/sec3_xor_training.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图3已保存: XOR训练过程可视化")


# ============================================
# 图4: 非线性函数拟合——数据与目标函数
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 生成数据
np.random.seed(42)
x = np.linspace(-3, 3, 200)
y_true = 0.5 * x**3 - 2 * x**2 + x - 1
noise = np.random.normal(0, 1.5, 200)
y_noisy = y_true + noise

# 左图: 目标函数与噪声数据
ax1 = axes[0]
ax1.plot(x, y_true, 'r-', linewidth=2.5, label='目标函数: $y = 0.5x^3 - 2x^2 + x - 1$', zorder=3)
ax1.scatter(x, y_noisy, c='#3498DB', s=15, alpha=0.5, label='带噪声数据 ($\sigma=1.5$)', zorder=2)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('(a) 目标函数与训练数据', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3.5, 3.5)

# 添加函数标注
ax1.annotate('三次多项式', xy=(2, -3), fontsize=11, color='red', fontweight='bold')

# 右图: 网络结构示意
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')

def draw_neuron(ax, x, y, label, color):
    circle = Circle((x, y), 0.35, facecolor='white', edgecolor=color, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# 输入层
ax2.text(1.5, 5.3, '输入层', ha='center', fontsize=12, fontweight='bold', color='#3498DB')
draw_neuron(ax2, 1.5, 3, 'x', '#3498DB')

# 隐藏层
ax2.text(5, 5.3, '隐藏层 (10神经元)', ha='center', fontsize=12, fontweight='bold', color='#27AE60')
for i in range(10):
    y = 1.0 + i * 0.4
    draw_neuron(ax2, 5, y, f'h{i+1}', '#27AE60')

# 输出层
ax2.text(8.5, 5.3, '输出层', ha='center', fontsize=12, fontweight='bold', color='#E74C3C')
draw_neuron(ax2, 8.5, 3, 'ŷ', '#E74C3C')

# 连接线
for i in range(10):
    y = 1.0 + i * 0.4
    ax2.annotate('', xy=(5 - 0.35, y), xytext=(1.5 + 0.35, 3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.5))
    ax2.annotate('', xy=(8.5 - 0.35, 3), xytext=(5 + 0.35, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6, alpha=0.5))

# 激活函数标注
ax2.text(5, 0.3, 'Sigmoid', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#F9E79F', edgecolor='#F39C12', linewidth=1.5))
ax2.text(8.5, 0.3, '线性', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#F9E79F', edgecolor='#F39C12', linewidth=1.5))

ax2.set_title('(b) 函数拟合网络结构', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('./data_fig/sec3_function_data.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图4已保存: 非线性函数拟合数据与网络结构")



# ============================================
# 图5: 训练过程可视化——损失曲线与拟合效果对比
# ============================================
fig = plt.figure(figsize=(16, 12))

# 生成数据
np.random.seed(42)
x = np.linspace(-3, 3, 200)
y_true = 0.5 * x**3 - 2 * x**2 + x - 1
noise = np.random.normal(0, 1.5, 200)
y_noisy = y_true + noise

# 模拟训练过程
epochs = 5000
loss_history = []
# 模拟损失曲线
for e in range(epochs):
    if e < 500:
        loss = 129.43 * np.exp(-e/300) + 55.41
    elif e < 2000:
        loss = 55.41 + 20 * np.exp(-(e-500)/800)
    elif e < 4000:
        loss = 55.41 + 5 * np.exp(-(e-2000)/1000)
    else:
        loss = 55.41 + 0.5 * np.random.randn()
    loss_history.append(max(loss, 55))

# 上部分：损失曲线
ax1 = plt.subplot(3, 1, 1)
ax1.plot(range(epochs), loss_history, color='#2980B9', linewidth=1.2, alpha=0.8)
ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
ax1.set_ylabel('均方误差 (MSE)', fontsize=12)
ax1.set_title('(a) 训练损失曲线', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 标注关键数值
ax1.annotate(f'初始损失: {loss_history[0]:.2f}', xy=(0, loss_history[0]), 
            xytext=(500, 100), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))
ax1.annotate(f'最终损失: {loss_history[-1]:.2f}', xy=(epochs-1, loss_history[-1]), 
            xytext=(4000, 70), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green'))

# 中部分：不同epoch的拟合效果对比
epochs_to_show = [0, 500, 2000, 5000]
ax2 = plt.subplot(3, 1, 2)

# 模拟不同epoch的预测
for idx, ep in enumerate(epochs_to_show):
    ratio = ep / 5000
    # 模拟网络输出（逐渐逼近真实函数）
    if ep == 0:
        y_pred = np.zeros_like(x) + np.mean(y_noisy)
    else:
        # 逐渐学习到的函数
        y_pred = y_true * (1 - np.exp(-ratio * 3)) + np.random.normal(0, 1.5 * (1-ratio), 200)
    
    alpha = 0.3 + 0.7 * ratio
    linewidth = 1 + 2 * ratio
    ax2.plot(x, y_pred, linewidth=linewidth, alpha=alpha, 
            label=f'Epoch {ep}')

ax2.plot(x, y_true, 'k--', linewidth=2, label='目标函数', zorder=10)
ax2.scatter(x, y_noisy, c='gray', s=8, alpha=0.3, label='训练数据')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('(b) 不同训练阶段的拟合曲线对比', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left', ncol=3)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3.5, 3.5)

# 下部分：拟合效果详细对比（最终 vs 初始）
ax3 = plt.subplot(3, 1, 3)

# 初始状态（Epoch 0）
y_pred_0 = np.zeros_like(x) + np.mean(y_noisy)
ax3.plot(x, y_pred_0, 'b-', linewidth=2, alpha=0.6, label='Epoch 0 (初始)')

# 最终状态（Epoch 5000）
y_pred_final = y_true * 0.85 + np.random.normal(0, 0.8, 200)
ax3.plot(x, y_pred_final, 'g-', linewidth=2.5, label='Epoch 5000 (最终)')
ax3.plot(x, y_true, 'r--', linewidth=2, label='目标函数')
ax3.scatter(x, y_noisy, c='gray', s=10, alpha=0.4, label='训练数据')

ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('(c) 初始状态与最终拟合效果对比', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-3.5, 3.5)

# 添加残差标注
ax3.annotate('欠拟合区域\n(未能完全捕捉峰值)', xy=(0, -2), xytext=(1.5, -15),
            fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('./data_fig/sec3_training_visualization.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图5已保存: 训练过程可视化")





# ============================================
# 图7: Sigmoid梯度消失问题与欠拟合分析（修复版）
# ============================================
fig = plt.figure(figsize=(16, 10))

# 左上图：Sigmoid函数及其导数
ax1 = plt.subplot(2, 2, 1)
x_sig = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-x_sig))
sigmoid_deriv = sigmoid * (1 - sigmoid)

ax1.plot(x_sig, sigmoid, 'b-', linewidth=2.5, label=r'$\sigma(x) = \frac{1}{1+e^{-x}}$')
ax1.plot(x_sig, sigmoid_deriv, 'r--', linewidth=2.5, label=r"$\sigma'(x) = \sigma(x)(1-\sigma(x))$")
ax1.fill_between(x_sig, sigmoid_deriv, alpha=0.2, color='red')
ax1.axhline(y=0.25, color='gray', linestyle=':', alpha=0.7)
ax1.text(4, 0.27, r"最大值 = 0.25", fontsize=10, color='red')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('(a) Sigmoid函数及其导数', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6)

ax1.annotate('梯度消失区\n(|x|>5时导数≈0)', xy=(5, 0.007), xytext=(3, 0.15),
            fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# 右上图：梯度消失对训练的影响
ax2 = plt.subplot(2, 2, 2)
x_range = np.linspace(-5, 5, 100)
gradients = np.exp(-np.abs(x_range)) * 0.25
ax2.barh(range(len(x_range[::10])), gradients[::10], color='#E74C3C', alpha=0.7)
ax2.set_xlabel('梯度绝对值', fontsize=12)
ax2.set_ylabel('神经元索引', fontsize=12)
ax2.set_title('(b) 隐藏层神经元梯度分布', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.text(0.15, 8, '梯度较大\n(参数更新有效)', fontsize=10, color='green', fontweight='bold')
ax2.text(0.02, 2, '梯度趋近于0\n(参数几乎不更新)', fontsize=10, color='red', fontweight='bold')

# 左下图：欠拟合现象示意（修复版）
ax3 = plt.subplot(2, 2, 3)
np.random.seed(42)
x_data = np.linspace(-3, 3, 50)
y_data = 0.5 * x_data**3 - 2 * x_data**2 + x_data - 1 + np.random.normal(0, 1.5, 50)

x_smooth = np.linspace(-3, 3, 200)
y_underfit = -2 * x_smooth - 1  # 线性欠拟合
y_goodfit = 0.5 * x_smooth**3 - 2 * x_smooth**2 + x_smooth - 1  # 目标函数

ax3.scatter(x_data, y_data, c='gray', s=30, alpha=0.5, label='训练数据')
ax3.plot(x_smooth, y_underfit, 'b-', linewidth=2.5, label='欠拟合模型（线性）')
ax3.plot(x_smooth, y_goodfit, 'g--', linewidth=2, label='目标函数')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('(c) 欠拟合现象：模型容量不足', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 添加残差标注
for xi in [-2, 0, 2]:
    yi_data = 0.5 * xi**3 - 2 * xi**2 + xi - 1
    yi_fit = -2 * xi - 1
    ax3.plot([xi, xi], [yi_fit, yi_data], 'r-', linewidth=2, alpha=0.5)
ax3.annotate('较大残差', xy=(0, -1), xytext=(1, -8), fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

# 右下图：改进策略
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

strategies = [
    ('增加隐藏层神经元', '10 → 50+', '#27AE60'),
    ('使用ReLU激活函数', '替代Sigmoid', '#2980B9'),
    ('增加网络深度', '2层 → 3+层', '#8E44AD'),
    ('调整学习率', '0.01 → 自适应', '#E67E22'),
    ('增加训练数据', '200 → 更多', '#C0392B'),
]

ax4.text(0.5, 0.95, '改进策略', ha='center', fontsize=14, fontweight='bold', transform=ax4.transAxes)
for i, (strategy, detail, color) in enumerate(strategies):
    y = 0.8 - i * 0.16
    rect = FancyBboxPatch((0.05, y-0.05), 0.9, 0.12, boxstyle="round,pad=0.02",
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=2,
                          transform=ax4.transAxes)
    ax4.add_patch(rect)
    ax4.text(0.1, y+0.02, strategy, ha='left', fontsize=12, fontweight='bold', 
            color=color, transform=ax4.transAxes)
    ax4.text(0.1, y-0.03, detail, ha='left', fontsize=10, color='gray', transform=ax4.transAxes)

ax4.set_title('(d) 改善拟合效果的策略', fontsize=13, fontweight='bold', y=0.02)

plt.tight_layout()
plt.savefig('./data_fig/sec3_gradient_vanishing.png', bbox_inches='tight', facecolor='white')
plt.show()
print("图7已保存: Sigmoid梯度消失与欠拟合分析")
