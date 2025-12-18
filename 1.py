import matplotlib.pyplot as plt
import numpy as np

# 设置画布
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 适配Ubuntu中文
plt.rcParams['axes.unicode_minus'] = False

# 定义MLP各层神经元数
layers = [65, 128, 64, 32, 2]
layer_names = ['输入层\n(65维特征)', '隐藏层1\n(128-ReLU)', '隐藏层2\n(64-ReLU)', '隐藏层3\n(32-ReLU)', '输出层\n(2-Softmax)']
colors = ['#4CAF50', '#2196F3', '#2196F3', '#2196F3', '#F44336']

# 绘制各层神经元
y_pos = np.arange(len(layers))
for i, (neurons, name, color) in enumerate(zip(layers, layer_names, colors)):
    # 绘制神经元（圆形）
    x = np.ones(neurons) * i
    y = np.linspace(0, neurons, neurons) if neurons > 2 else [0.5, 1.5]
    plt.scatter(x, y, s=60, c=color, alpha=0.8, label=name if i == 0 or i == len(layers)-1 else "")
    
    # 标注层名称
    plt.text(i, -5, name, ha='center', va='center', fontsize=10, fontweight='bold')

# 绘制层间连接
for i in range(len(layers)-1):
    # 取前一层和后一层的神经元坐标
    x1 = np.ones(layers[i]) * i
    y1 = np.linspace(0, layers[i], layers[i]) if layers[i] > 2 else [0.5, 1.5]
    x2 = np.ones(layers[i+1]) * (i+1)
    y2 = np.linspace(0, layers[i+1], layers[i+1]) if layers[i+1] > 2 else [0.5, 1.5]
    
    # 绘制连接线条（简化：随机选部分连接，避免画面过密）
    for xi, yi in zip(x1[::max(1, layers[i]//20)], y1[::max(1, layers[i]//20)]):
        for xj, yj in zip(x2[::max(1, layers[i+1]//20)], y2[::max(1, layers[i+1]//20)]):
            plt.plot([xi, xj], [yi, yj], c='#CCCCCC', alpha=0.2, linewidth=0.5)

# 调整画布范围和样式
plt.xlim(-0.5, len(layers)-0.5)
plt.ylim(-10, max(layers)+5)
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.title('轨迹分类任务 MLP 层结构示意图', fontsize=14, fontweight='bold', pad=20)

# 保存图片（可选）
plt.savefig('mlp_structure.png', dpi=300, bbox_inches='tight')
plt.show()