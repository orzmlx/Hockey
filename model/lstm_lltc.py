import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv("../data/PRSA_data_2010.1.1-2014.12.31.csv")

# 时间字段构建
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df.set_index("datetime", inplace=True)

# 删除无用字段
df = df.drop(columns=["No", "year", "month", "day", "hour", "cbwd"])

# 去除缺失
df = df.dropna(subset=['pm2.5'])

# 仅使用 PM2.5 预测
data = df[["pm2.5"]].values

# 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 构建时间序列样本（滑动窗口）
def create_dataset(dataset, look_back=24):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

look_back = 24
X, y = create_dataset(data_scaled, look_back)

# 拆分训练与测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=64)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out
class LNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h):
        dh = torch.tanh(x @ self.W + h @ self.U) - h
        h = h + dh / self.tau
        return h

class LNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.cell = LNNCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.cell.U.size(0)).to(x.device)
        for t in range(x.size(1)):
            h = self.cell(x[:, t, :], h)
        return self.linear(h)


def train_model(model, epochs=20):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            output = model(xb)
            loss = criterion(output.squeeze(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Train Loss = {loss.item():.6f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred = model(xb)
            preds.append(pred.cpu())
    return torch.cat(preds).numpy()
import matplotlib.pyplot as plt
from torchsummary import summary


# LSTM
lstm_model = LSTMModel()
#lstm_preds = train_model(lstm_model)

# LNN
lnn_model = LNNModel()
#lnn_preds = train_model(lnn_model)

summary(lstm_model, input_size=(24,1))  # 替
summary(lnn_model, input_size=( 24, 1))  # 替

# 还原原始尺度
true = scaler.inverse_transform(y_test)
lstm_inv = scaler.inverse_transform(lstm_preds)
lnn_inv = scaler.inverse_transform(lnn_preds)

# 可视化
plt.figure(figsize=(12,6))
plt.plot(y_test[:300], label="True", linewidth=2)
plt.plot(lstm_preds[:300], label="LSTM", linestyle="--")
plt.plot(lnn_preds[:300], label="LNN", linestyle=":")
plt.legend()
plt.title("PM2.5 Forecasting: LSTM vs LNN")
plt.xlabel("Time Steps")
plt.ylabel("PM2.5")
plt.grid()
plt.tight_layout()
plt.show()
