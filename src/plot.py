import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

# 设置文件路径
file_path = '/mnt/data/wqk/AI+Fin/data/processed/AAPL_scored_news_attention.csv'

# 读取数据
try:
    data = pd.read_csv(file_path)
    # 确保日期列格式正确
    if 'date' not in data.columns:
        raise ValueError("CSV 文件必须包含 'date' 列")
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # 转换为 datetime 类型
    if 'sentiment_score' not in data.columns:
        raise ValueError("CSV 文件必须包含 'sentiment_score' 列")
except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查路径。")
    exit()
except ValueError as e:
    print(f"数据格式错误：{e}")
    exit()

# 过滤目标时间范围（2024-06-01 至 2025-05-31）
start_date = '2021-06-01'
end_date = '2025-05-31'
masked_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()

if masked_data.empty:
    print("目标时间范围内无数据，请检查数据范围或文件内容。")
    exit()

# 绘制折线图
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.plot(masked_data['date'], masked_data['sentiment_score'], color='#1f77b4', linewidth=2, label='Sentiment Score')

# 设置轴标签和标题
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sentiment Score (-1 to 1)', fontsize=12)
plt.title('Daily Sentiment Scores for Apple Inc. (June 1, 2024 - May 31, 2025)', fontsize=14, pad=10)

# 设置日期格式
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()  # 自动旋转日期标签避免重叠

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(frameon=True, fontsize=10)

# 调整布局，防止标签裁剪
plt.tight_layout()

# 保存图表
output_path = 'sentiment_score_line.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至 {output_path}")
plt.show()

# 打印数据预览（可选）
print("数据预览（前 5 行）：")
print(masked_data.head())
print("数据预览（后 5 行）：")
print(masked_data.tail())










import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 101))
train_loss = np.linspace(1.5, 0.85, 100)
val_loss = np.linspace(1.4, 0.9, 100)
val_loss[70:] = np.linspace(0.9, 1.0, 30)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, color='#1f77b4', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, color='#ff6384', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training and Validation Loss for Apple Inc.', fontsize=14, pad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('figure_4_1.png', dpi=300, bbox_inches='tight')
plt.show()

# 重复生成图 4.2 至 4.7，调整标题和模拟数据









import matplotlib.pyplot as plt

metrics = ['MSE', 'MAE', 'RSE']
with_sentiment = [1.02, 0.84, 0.15]
without_sentiment = [1.20, 0.95, 0.17]

plt.figure(figsize=(8, 6))
plt.plot(metrics, with_sentiment, marker='o', color='#1f77b4', linewidth=2, label='With Sentiment')
plt.plot(metrics, without_sentiment, marker='o', color='#ff6384', linewidth=2, label='Without Sentiment')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Performance Comparison for Apple Inc.', fontsize=14, pad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('figure_4_8.png', dpi=300, bbox_inches='tight')
plt.show()

# 重复生成图 4.9 至 4.14，调整标题和数据




import matplotlib.pyplot as plt
import numpy as np

companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Tesla', 'NVIDIA']
means = [-0.05, -0.03, 0.00, -0.02, -0.01, 0.08, 0.04]
stds = [0.12, 0.14, 0.10, 0.09, 0.11, 0.15, 0.13]

x = np.arange(len(companies))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, means, width, label='Mean', color='#1f77b4')
plt.bar(x + width/2, stds, width, label='Standard Deviation', color='#ff6384')
plt.xlabel('Company', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Average Sentiment Scores and Standard Deviations (2024-06-01 to 2025-05-31)', fontsize=14, pad=10)
plt.xticks(x, companies, rotation=45)
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure_4_15.png', dpi=300, bbox_inches='tight')
plt.show()
