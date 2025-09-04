import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset

# 文件列表
file_paths = [
    '105.AAPL_苹果_with_sentiment.csv',
    '105.AMZN_亚马逊_with_sentiment.csv',
    '105.GOOGL_谷歌_with_sentiment.csv',
    '105.META_Meta Platforms Inc_with_sentiment.csv',
    '105.MSFT_微软_with_sentiment.csv',
    '105.NVDA_英伟达_with_sentiment.csv',
    '105.TSLA_特斯拉_with_sentiment.csv'
]

for file_path in file_paths:
    company = file_path.split('_')[1].split('_')[0]  # 如 'AAPL'

    df = pd.read_csv(file_path, encoding='utf-8-sig')  # 使用utf-8-sig处理BOM

    df = df.dropna()

    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d')

    features = ['开盘价', '收盘价', '最高价', '最低价', '成交量(手)', '成交额(元)', '振幅(%)', '涨跌幅(%)', '涨跌额(美元)', '换手率(%)', '情感分数']
    data = df[features].values  # 提取所有数值列的值，确保使用全部数据

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 参数设置
    # 定义时间步长
    time_step = 60  # 使用过去60天的数据预测下一天

    # 创建序列数据（X: 过去time_step天的所有特征，y: 下一天的收盘价）
    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step)])
            y.append(data[i + time_step, 1]) 
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, time_step)

    # 分割训练集和测试集，这里使用前80%作为训练，后20%作为测试（测试集兼作验证集）
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义GRU模型
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)  # 添加dropout防止过拟合
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    # 模型参数设置
    input_size = len(features)  # 自动适应所有特征（11个）
    hidden_size = 100  # 增大隐藏层大小，以处理更多特征
    num_layers = 2
    output_size = 1  # 预测收盘价

    model = GRUModel(input_size, hidden_size, num_layers, output_size)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率，提高稳定性

    # 训练模型，并收集损失值用于早停
    num_epochs = 200  # 最大epochs，实际可能因早停而提前结束
    patience = 10  # 早停耐心值：连续patience个epoch验证损失不降则停止
    min_delta = 0.0001  # 损失下降的最小阈值
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1
        avg_train_loss = epoch_train_loss / num_batches
        
        # 验证模式（使用测试集作为验证）
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(test_loader)
        
        # 早停检查
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # 测试集上的评估指标
    # 先获取测试集预测（反归一化）
    with torch.no_grad():
        test_pred_scaled = model(X_test).detach().numpy()
        test_pred_full = np.zeros((len(test_pred_scaled), len(features)))
        test_pred_full[:, 1] = test_pred_scaled[:, 0]  # 填充收盘价位置
        test_pred = scaler.inverse_transform(test_pred_full)[:, 1]

    # 测试集实际值（从原数据中提取收盘价）
    test_actual = df['收盘价'].values[time_step + split:]

    # 计算MSE, MAE, RMSE
    mse = np.mean((test_pred - test_actual) ** 2)
    mae = np.mean(np.abs(test_pred - test_actual))
    rmse = np.sqrt(mse)

    #特征数，n 为测试样本数）
    n = len(test_actual)
    p = len(features) # 添加 RSE 计算（p 为简单近似，使用特征数作为 p（可根据需要调整为模型总参数数）
    sse = mse * n  # Sum of Squared Errors
    rse = np.sqrt(sse / (n - p - 1)) if n > p + 1 else rmse  # 避免除零或负

    # 计算Directional Accuracy（使用前一天实际值作为基准，提高交易可操作性）
    direction_correct = 0
    total = len(test_actual) - 1  # 从第二个开始计算变化
    for i in range(1, len(test_actual)):
        actual_delta = test_actual[i] - test_actual[i-1]
        pred_delta = test_pred[i] - test_actual[i-1]
        if np.sign(actual_delta) == np.sign(pred_delta):
            direction_correct += 1
    directional_accuracy = direction_correct / total if total > 0 else 0

    # 打印评估结果（仅指标）
    print(f"Company: {company}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RSE: {rse:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f} ({direction_correct}/{total})")
    print("---")  # 分隔符
