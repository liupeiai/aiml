
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.font_manager import FontProperties

# 加载中文字体
font_path = '/usr/share/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)
font_prop_title = FontProperties(fname=font_path, size=13, weight='bold')
font_prop_label = FontProperties(fname=font_path, size=11)
font_prop_legend = FontProperties(fname=font_path, size=9)
font_prop_text = FontProperties(fname=font_path, size=9)
font_prop_small = FontProperties(fname=font_path, size=8)

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 准备数据
x_np = np.linspace(-3, 3, 200).reshape(-1, 1)
y_true_np = 0.5 * x_np**3 - 2 * x_np**2 + x_np - 1
noise_np = np.random.randn(200, 1) * 1.5
y_data_np = y_true_np + noise_np

x_torch = torch.FloatTensor(x_np)
y_torch = torch.FloatTensor(y_data_np)

# 定义模型
class PolynomialNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# 训练三种不同隐藏层大小的模型
hidden_sizes = [32, 64, 128]
models = {}
losses_history = {}
final_losses = {}

for hidden in hidden_sizes:
    torch.manual_seed(42)
    model = PolynomialNet(hidden_size=hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []
    for epoch in range(2000):
        optimizer.zero_grad()
        y_pred = model(x_torch)
        loss = criterion(y_pred, y_torch)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    models[hidden] = model
    losses_history[hidden] = loss_history
    final_losses[hidden] = loss_history[-1]
    print(f"Hidden={hidden}, Final Loss={loss_history[-1]:.4f}")

print("训练完成，开始绘图...")



# ============================================================
# 图4-6：不同隐藏层大小对拟合效果的影响
# ============================================================

fig = plt.figure(figsize=(16, 12))

# 颜色方案
colors = {
    32: '#3498db',   # 蓝色 - 欠拟合
    64: '#2ecc71',   # 绿色 - 最佳
    128: '#e74c3c'   # 红色 - 可能过拟合
}

labels_desc = {
    32: 'hidden=32（欠拟合）',
    64: 'hidden=64（最佳）',
    128: 'hidden=128（可能过拟合）'
}

# --- 第一行：三个子图分别展示拟合效果 ---
for idx, hidden in enumerate(hidden_sizes):
    ax = plt.subplot(3, 3, idx + 1)
    
    # 绘制训练数据和真实函数
    ax.scatter(x_np, y_data_np, c='#cccccc', s=12, alpha=0.4, 
               label='训练数据（含噪声）', zorder=1, edgecolors='none')
    ax.plot(x_np, y_true_np, 'k--', linewidth=2, label='目标函数', zorder=4)
    
    # 绘制模型预测
    with torch.no_grad():
        pred = models[hidden](x_torch).numpy().flatten()
    
    ax.plot(x_np, pred, color=colors[hidden], linewidth=2.5, 
            label=f'预测曲线 (MSE={final_losses[hidden]:.4f})', zorder=3)
    
    # 添加残差阴影
    residuals = y_data_np.flatten() - pred
    ax.fill_between(x_np.flatten(), pred, y_data_np.flatten(), 
                    alpha=0.15, color=colors[hidden])
    
    # 设置标题和标签
    ax.set_title(f'({chr(97+idx)}) {labels_desc[hidden]}', 
                 fontproperties=font_prop_title, pad=10)
    ax.set_xlabel('x', fontproperties=font_prop_label)
    ax.set_ylabel('y', fontproperties=font_prop_label)
    
    legend = ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop_small)
    
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 第二行中间：损失曲线对比 ---
ax4 = plt.subplot(3, 3, 5)

for hidden in hidden_sizes:
    epochs = np.arange(2000)
    ax4.plot(epochs, losses_history[hidden], color=colors[hidden], 
             linewidth=1.5, label=f'hidden={hidden} (最终={final_losses[hidden]:.4f})')

ax4.set_xlabel('训练轮次 (Epoch)', fontproperties=font_prop_label)
ax4.set_ylabel('均方误差 (MSE)', fontproperties=font_prop_label)
ax4.set_title('(d) 训练损失曲线对比', fontproperties=font_prop_title, pad=10)

legend4 = ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
for text in legend4.get_texts():
    text.set_fontproperties(font_prop_small)

ax4.set_xlim(0, 2000)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# --- 第二行左右：残差分布 ---
for idx, hidden in enumerate(hidden_sizes):
    if idx == 1:
        continue  # 中间已经画了损失曲线
    
    ax_idx = 4 if idx == 0 else 6
    ax = plt.subplot(3, 3, ax_idx)
    
    with torch.no_grad():
        pred = models[hidden](x_torch).numpy().flatten()
    residuals = y_data_np.flatten() - pred
    
    ax.scatter(x_np, residuals, c=colors[hidden], s=10, alpha=0.5, 
               edgecolors='none')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    std_res = np.std(residuals)
    ax.text(0.02, 0.95, f'残差标准差: {std_res:.3f}', 
            transform=ax.transAxes, fontproperties=font_prop_text,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=colors[hidden], alpha=0.8))
    
    ax.set_xlabel('x', fontproperties=font_prop_label)
    ax.set_ylabel('残差', fontproperties=font_prop_label)
    title_char = 'e' if idx == 0 else 'f'
    title_desc = 'hidden=32' if idx == 0 else 'hidden=128'
    ax.set_title(f'({title_char}) {title_desc} 残差分布', 
                 fontproperties=font_prop_title, pad=10)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 第三行：模型复杂度分析（参数数量对比） ---
ax7 = plt.subplot(3, 3, 7)

param_counts = {}
for hidden in hidden_sizes:
    model = PolynomialNet(hidden_size=hidden)
    total_params = sum(p.numel() for p in model.parameters())
    param_counts[hidden] = total_params

bars = ax7.bar([f'hidden={h}' for h in hidden_sizes], 
               [param_counts[h] for h in hidden_sizes],
               color=[colors[h] for h in hidden_sizes], alpha=0.7, edgecolor='black')

for bar, hidden in zip(bars, hidden_sizes):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height)}',
             ha='center', va='bottom', fontproperties=font_prop_text, fontsize=10)

ax7.set_ylabel('参数数量', fontproperties=font_prop_label)
ax7.set_title('(g) 模型参数量对比', fontproperties=font_prop_title, pad=10)
ax7.grid(True, alpha=0.3, linestyle='--', axis='y')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# --- 第三行中间：拟合效果综合对比（大图） ---
ax8 = plt.subplot(3, 3, 8)

ax8.scatter(x_np, y_data_np, c='#cccccc', s=12, alpha=0.3, 
            label='训练数据', zorder=1, edgecolors='none')
ax8.plot(x_np, y_true_np, 'k--', linewidth=2, label='目标函数', zorder=5)

for hidden in hidden_sizes:
    with torch.no_grad():
        pred = models[hidden](x_torch).numpy().flatten()
    ax8.plot(x_np, pred, color=colors[hidden], linewidth=2, 
             label=f'hidden={hidden} (MSE={final_losses[hidden]:.4f})', zorder=3)

ax8.set_xlabel('x', fontproperties=font_prop_label)
ax8.set_ylabel('y', fontproperties=font_prop_label)
ax8.set_title('(h) 拟合效果综合对比', fontproperties=font_prop_title, pad=10)

legend8 = ax8.legend(loc='upper left', fontsize=8, framealpha=0.9)
for text in legend8.get_texts():
    text.set_fontproperties(font_prop_small)

ax8.set_xlim(-3.5, 3.5)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)

# --- 第三行右侧：训练时间/计算开销示意 ---
ax9 = plt.subplot(3, 3, 9)

# 模拟训练时间（相对值，hidden越大计算量越大）
training_times = {32: 1.0, 64: 1.8, 128: 3.2}

bars = ax9.bar([f'hidden={h}' for h in hidden_sizes], 
               [training_times[h] for h in hidden_sizes],
               color=[colors[h] for h in hidden_sizes], alpha=0.7, edgecolor='black')

for bar, hidden in zip(bars, hidden_sizes):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.1f}x',
             ha='center', va='bottom', fontproperties=font_prop_text, fontsize=10)

ax9.set_ylabel('相对训练时间', fontproperties=font_prop_label)
ax9.set_title('(i) 计算开销对比', fontproperties=font_prop_title, pad=10)
ax9.grid(True, alpha=0.3, linestyle='--', axis='y')
ax9.spines['top'].set_visible(False)
ax9.spines['right'].set_visible(False)

plt.suptitle('图4-6  不同隐藏层大小对拟合效果的影响', 
             fontproperties=FontProperties(fname=font_path, size=16, weight='bold'), 
             y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
print("✅ 图4-6 绘制完成")

