import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import math

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 加载数据集
data = pd.read_csv('1.csv', parse_dates=['日期'], encoding='utf-8')

# 数据预处理
# 将日期设为索引
data.set_index('日期', inplace=True)

# 选择特征
features = ['开盘价', '收盘价', '最高价', '最低价', '成交量(手)', '成交额(元)', '振幅(%)', 
            '涨跌幅(%)', '涨跌额(美元)', '换手率(%)', '情感分数']
X = data[features].values

# 创建目标变量：未来 5 天的收盘价
y = []
for i in range(len(data) - 5):
    y.append(data['收盘价'].values[i+1:i+6])
y = np.array(y)

# 从 X 中移除最后 5 行以对齐 y
X = X[:-5]

# 标准化特征
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 标准化目标
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# 分为训练集和验证集（80% 训练，20% 验证）
train_size = int(len(X_scaled) * 0.8)
X_train, X_val = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

# 转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train).reshape(-1, 1, X_train.shape[1])  # [样本, 时间步=1, 特征]
X_val = torch.FloatTensor(X_val).reshape(-1, 1, X_val.shape[1])
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)

# 定义增强的 AMISA 模型
class AMISA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, d_model=64, lstm_hidden=128):
        super(AMISA, self).__init__()
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        # LSTM 层
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # x: [batch, timesteps=1, input_dim]
        x = self.input_projection(x)  # [batch, timesteps=1, d_model]
        x, _ = self.lstm(x)  # [batch, timesteps=1, lstm_hidden]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(attn_output + x)  # 残差连接
        x = x.squeeze(1)  # 移除时间步维度
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = AMISA(input_dim=X_train.shape[2], output_dim=5, num_heads=4, d_model=64, lstm_hidden=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率

# 早停参数
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# 训练循环
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # 验证损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    model.train()
    
    print(f'第 [{epoch+1}/{num_epochs}] 轮, 训练损失: {loss.item():.4f}, 验证损失: {val_loss.item():.4f}')
    
    # 早停检查
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停：第 {epoch+1} 轮后停止")
            break

# 加载最佳模型
model.load_state_dict(best_model_state)

# 在验证集上评估
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_val).numpy()
    y_val_numpy = y_val.numpy()

# 计算指标（在标准化尺度下）
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    
    # 方向准确率
    direction_correct = 0
    total = 0
    for i in range(1, len(y_true)):
        actual_direction = y_true[i] - y_true[i-1]
        pred_direction = y_pred[i] - y_pred[i-1]
        if (actual_direction * pred_direction).mean() > 0:  # 方向一致
            direction_correct += 1
        total += 1
    directional_accuracy = direction_correct / total if total > 0 else 0
    
    return mse, mae, rmse, directional_accuracy

# 计算标准化尺度下的指标
mse, mae, rmse, directional_accuracy = calculate_metrics(y_val_numpy, y_pred_scaled)

# 反标准化预测值和实际值（仅用于调试）
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_val_actual = scaler_y.inverse_transform(y_val_numpy)
mse_original = mean_squared_error(y_val_actual, y_pred)

# 打印验证集指标
print("\n验证集指标（标准化尺度）：")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"方向准确率: {directional_accuracy:.4f}")
print(f"\n反标准化尺度下的 MSE（仅供参考）: {mse_original:.4f}")
