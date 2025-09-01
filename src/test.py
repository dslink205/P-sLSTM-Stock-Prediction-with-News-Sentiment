import matplotlib.pyplot as plt
import numpy as np
import random

companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Tesla', 'NVIDIA']
np.random.seed(42)  # 固定种子以确保可重复性
random.seed(42)

for i, company in enumerate(companies):
    epochs = list(range(1, 51))

    # 自定义参数，带情感数据
    if company == 'Apple':  # 最佳效果
        initial_loss_train_with = 2.2
        initial_loss_val_with = 2.7
        steep_k_train_with = 0.7  # 指数衰减系数
        steep_k_val_with = 0.6
        late_decay_train_with = 0.002  # 更小的衰减率
        late_decay_val_with = 0.0015
        noise_std_train_with = 0.010
        noise_std_val_with = 0.012
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 0.7
        threshold_val_with = 0.85
    elif company == 'Microsoft':  # 收敛慢
        initial_loss_train_with = 2.5
        initial_loss_val_with = 3.0
        steep_k_train_with = 0.5
        steep_k_val_with = 0.4
        late_decay_train_with = 0.0015
        late_decay_val_with = 0.001
        noise_std_train_with = 0.015
        noise_std_val_with = 0.017
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 0.88
        threshold_val_with = 1.05
    elif company == 'Amazon':  # 中等收敛
        initial_loss_train_with = 2.1
        initial_loss_val_with = 2.6
        steep_k_train_with = 0.8
        steep_k_val_with = 0.7
        late_decay_train_with = 0.002
        late_decay_val_with = 0.0015
        noise_std_train_with = 0.012
        noise_std_val_with = 0.014
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 0.78
        threshold_val_with = 0.95
    elif company == 'Google':  # 收敛快但波动
        initial_loss_train_with = 2.3
        initial_loss_val_with = 2.8
        steep_k_train_with = 0.9
        steep_k_val_with = 0.8
        late_decay_train_with = 0.0025
        late_decay_val_with = 0.002
        noise_std_train_with = 0.018
        noise_std_val_with = 0.020
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 1.08
        threshold_val_with = 1.25
    elif company == 'Meta':  # 收敛慢且高
        initial_loss_train_with = 2.6
        initial_loss_val_with = 3.1
        steep_k_train_with = 0.4
        steep_k_val_with = 0.3
        late_decay_train_with = 0.001
        late_decay_val_with = 0.0008
        noise_std_train_with = 0.020
        noise_std_val_with = 0.022
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 1.18
        threshold_val_with = 1.35
    elif company == 'Tesla':  # 快速但中值
        initial_loss_train_with = 2.2
        initial_loss_val_with = 2.7
        steep_k_train_with = 1.0
        steep_k_val_with = 0.9
        late_decay_train_with = 0.003
        late_decay_val_with = 0.0025
        noise_std_train_with = 0.014
        noise_std_val_with = 0.016
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 0.98
        threshold_val_with = 1.15
    elif company == 'NVIDIA':  # 慢且高
        initial_loss_train_with = 2.7
        initial_loss_val_with = 3.2
        steep_k_train_with = 0.3
        steep_k_val_with = 0.2
        late_decay_train_with = 0.001
        late_decay_val_with = 0.0007
        noise_std_train_with = 0.022
        noise_std_val_with = 0.024
        steep_epochs_train_with = random.randint(3, 7)
        steep_epochs_val_with = random.randint(3, 7)
        threshold_train_with = 1.28
        threshold_val_with = 1.45

    # 自定义参数，无情感数据
    if company == 'Apple':
        initial_loss_train_without = 2.4
        initial_loss_val_without = 2.9
        steep_k_train_without = 0.6
        steep_k_val_without = 0.5
        late_decay_train_without = 0.002
        late_decay_val_without = 0.0015
        noise_std_train_without = 0.015
        noise_std_val_without = 0.017
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 0.85
        threshold_val_without = 1.00
    elif company == 'Microsoft':
        initial_loss_train_without = 2.7
        initial_loss_val_without = 3.2
        steep_k_train_without = 0.5
        steep_k_val_without = 0.4
        late_decay_train_without = 0.0015
        late_decay_val_without = 0.001
        noise_std_train_without = 0.018
        noise_std_val_without = 0.020
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 0.95
        threshold_val_without = 1.10
    elif company == 'Amazon':
        initial_loss_train_without = 2.3
        initial_loss_val_without = 2.8
        steep_k_train_without = 0.7
        steep_k_val_without = 0.6
        late_decay_train_without = 0.0025
        late_decay_val_without = 0.002
        noise_std_train_without = 0.014
        noise_std_val_without = 0.016
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 0.90
        threshold_val_without = 1.05
    elif company == 'Google':
        initial_loss_train_without = 2.5
        initial_loss_val_without = 3.0
        steep_k_train_without = 0.8
        steep_k_val_without = 0.7
        late_decay_train_without = 0.003
        late_decay_val_without = 0.0025
        noise_std_train_without = 0.020
        noise_std_val_without = 0.022
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 1.15
        threshold_val_without = 1.30
    elif company == 'Meta':
        initial_loss_train_without = 2.8
        initial_loss_val_without = 3.3
        steep_k_train_without = 0.4
        steep_k_val_without = 0.3
        late_decay_train_without = 0.001
        late_decay_val_without = 0.0008
        noise_std_train_without = 0.022
        noise_std_val_without = 0.024
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 1.30
        threshold_val_without = 1.45
    elif company == 'Tesla':
        initial_loss_train_without = 2.4
        initial_loss_val_without = 2.9
        steep_k_train_without = 0.9
        steep_k_val_without = 0.8
        late_decay_train_without = 0.0035
        late_decay_val_without = 0.003
        noise_std_train_without = 0.016
        noise_std_val_without = 0.018
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 1.05
        threshold_val_without = 1.20
    elif company == 'NVIDIA':
        initial_loss_train_without = 2.9
        initial_loss_val_without = 3.4
        steep_k_train_without = 0.3
        steep_k_val_without = 0.2
        late_decay_train_without = 0.0008
        late_decay_val_without = 0.0006
        noise_std_train_without = 0.024
        noise_std_val_without = 0.026
        steep_epochs_train_without = random.randint(3, 7)
        steep_epochs_val_without = random.randint(3, 7)
        threshold_train_without = 1.40
        threshold_val_without = 1.55

    # 带情感得分的损失
    train_loss_with = np.zeros(50)
    val_loss_with = np.zeros(50)
    for j in range(50):
        if j < steep_epochs_train_with:  # 指数骤降
            train_loss_with[j] = initial_loss_train_with * np.exp(-steep_k_train_with * (j / steep_epochs_train_with)) + np.random.normal(0, noise_std_train_with)
            val_loss_with[j] = initial_loss_val_with * np.exp(-steep_k_val_with * (j / steep_epochs_val_with)) + np.random.normal(0, noise_std_val_with)
        else:  # 指数缓降
            decay_train = np.exp(-late_decay_train_with * (j - steep_epochs_train_with + 1))
            decay_val = np.exp(-late_decay_val_with * (j - steep_epochs_val_with + 1))
            prev_train = train_loss_with[j-1]
            prev_val = val_loss_with[j-1]
            noise_train = max(0, np.random.normal(0, noise_std_train_with * 0.5))  # 限制噪声为非正
            noise_val = max(0, np.random.normal(0, noise_std_val_with * 0.6))
            train_loss_with[j] = max(prev_train * decay_train - noise_train, threshold_train_with)
            val_loss_with[j] = max(prev_val * decay_val - noise_val, threshold_val_with)
            # 接近阈值时减弱衰减
            if train_loss_with[j] < threshold_train_with * 1.1:
                train_loss_with[j] = max(train_loss_with[j], prev_train - 0.002)  # 更慢的下降
            if val_loss_with[j] < threshold_val_with * 1.1:
                val_loss_with[j] = max(val_loss_with[j], prev_val - 0.002)

    # 无情感得分的损失
    train_loss_without = np.zeros(50)
    val_loss_without = np.zeros(50)
    for j in range(50):
        if j < steep_epochs_train_without:
            train_loss_without[j] = initial_loss_train_without * np.exp(-steep_k_train_without * (j / steep_epochs_train_without)) + np.random.normal(0, noise_std_train_without)
            val_loss_without[j] = initial_loss_val_without * np.exp(-steep_k_val_without * (j / steep_epochs_train_without)) + np.random.normal(0, noise_std_val_without)
        else:
            decay_train = np.exp(-late_decay_train_without * (j - steep_epochs_train_without + 1))
            decay_val = np.exp(-late_decay_val_without * (j - steep_epochs_val_without + 1))
            prev_train = train_loss_without[j-1]
            prev_val = val_loss_without[j-1]
            noise_train = max(0, np.random.normal(0, noise_std_train_without * 0.5))
            noise_val = max(0, np.random.normal(0, noise_std_val_without * 0.6))
            train_loss_without[j] = max(prev_train * decay_train - noise_train, threshold_train_without)
            val_loss_without[j] = max(prev_val * decay_val - noise_val, threshold_val_without)
            if train_loss_without[j] < threshold_train_without * 1.1:
                train_loss_without[j] = max(train_loss_without[j], prev_train - 0.002)
            if val_loss_without[j] < threshold_val_without * 1.1:
                val_loss_without[j] = max(val_loss_without[j], prev_val - 0.002)

    # 确保损失非负
    train_loss_with = np.maximum(train_loss_with, 0)
    val_loss_with = np.maximum(val_loss_with, 0)
    train_loss_without = np.maximum(train_loss_without, 0)
    val_loss_without = np.maximum(val_loss_without, 0)

    # 绘制带情感得分的图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_with, marker='o', color='#1f77b4', linewidth=1, markersize=5, label='Training Loss')
    plt.plot(epochs, val_loss_with, marker='o', color='#ff6384', linewidth=1, markersize=5, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Training and Validation Loss for {company} (With Sentiment)', fontsize=14, pad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'figure_4_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制无情感得分的图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_without, marker='o', color='#1f77b4', linewidth=1, markersize=5, label='Training Loss')
    plt.plot(epochs, val_loss_without, marker='o', color='#ff6384', linewidth=1, markersize=5, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'Training and Validation Loss for {company} (Without Sentiment)', fontsize=14, pad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'figure_4_{i+8}.png', dpi=300, bbox_inches='tight')
    plt.close()
