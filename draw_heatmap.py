import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 引入你的项目依赖
from DSTLF import ParquetDataset, collate_fn
import models.DeepTTE as DeepTTE

def plot_attention_heatmap(model_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据 (必须使用 mode='test'，关闭随机噪声增强)
    dataset = ParquetDataset(file_path=data_path, normalize=True)
    # batch_size=1 方便我们一个一个提取样本
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 2. 初始化模型 (确保这里的参数和你训练 best_model.pth 时一模一样！)
    model = DeepTTE.Net(
        num_classes=2, 
        num_filter=32, 
        hidden_size=48, 
        num_fc_layers=1, 
        dropout_p=0.5
    )
    
    # 3. 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # 开启测试模式

    # 用于标记是否找到了合适的样本
    found_hc, found_mci = False, False

    # 创建一个 1行2列 的画布
    plt.figure(figsize=(14, 6))

    print("正在测试集中寻找典型的正确预测样本...")
    with torch.no_grad():
        for attr, traj, labels, meta_ids in loader:
            # 如果两个都找到了，就停止搜索
            if found_hc and found_mci:
                break

            label = labels.item()
            # 如果这个类别已经找到了，跳过
            if label == 0 and found_hc: continue
            if label == 1 and found_mci: continue

            # 数据上设备
            attr_dev = {k: v.to(device) for k, v in attr.items()}
            traj_dev = {k: v.to(device) for k, v in traj.items()}

            # 🔴 前向传播，开启 return_attention=True
            logits, alpha = model(attr_dev, traj_dev, return_attention=True)
            pred = torch.argmax(logits, dim=1).item()

            # 为了热力图有说服力，我们只画“模型预测完全正确”的样本
            if pred != label: 
                continue

            # 提取掩码，获取该样本真实的有效长度 (去掉 padding 的 0)
            mask = traj['mask'][0].numpy()
            valid_len = int(mask.sum())

            # 提取轨迹坐标 (注意：这是归一化后的坐标，但不影响形状的可视化)
            ex = traj['ex'][0].numpy()[:valid_len]
            ey = traj['ey'][0].numpy()[:valid_len]
            
            # 提取对应的 Attention 权重
            attn_weights = alpha[0].cpu().numpy()[:valid_len]

            # ---------------- 画图逻辑 ----------------
            # 指定画在左边(HC)还是右边(MCI)
            ax = plt.subplot(1, 2, 1 if label == 0 else 2)
            
            # 1. 先画一条浅灰色的底线，勾勒出轨迹的轮廓
            ax.plot(ex, ey, color='gray', alpha=0.3, linewidth=1.5, zorder=1)
            
            # 2. 在坐标点上画散点，用 attention 权重作为颜色映射
            # cmap='jet' 是经典的红蓝热力图 (红高蓝低)
            sc = ax.scatter(ex, ey, c=attn_weights, cmap='jet', s=30, alpha=0.9, zorder=2)
            
            # 增加颜色条
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)

            # 设置标题
            title = f"Healthy Control (HC)" if label == 0 else f"MCI Patient"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('equal') # 保持XY比例一致，防止图形拉伸变形
            ax.axis('off')   # 去除丑陋的坐标轴边框

            # 标记已找到
            if label == 0: found_hc = True
            if label == 1: found_mci = True

    plt.tight_layout()
    # 存为高清图片，供论文使用
    plt.savefig("attention_heatmap.png", dpi=300, bbox_inches='tight')
    print("✅ 绘图完成！已保存为 attention_heatmap.png")
    plt.show()

if __name__ == "__main__":
    # 请确保路径正确
    MODEL_PATH = "best_model.pth" 
    DATA_PATH = "data2.parquet"
    plot_attention_heatmap(MODEL_PATH, DATA_PATH)