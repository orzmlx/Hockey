import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built with PyTorch: {torch.backends.mps.is_built()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):


        # if mps is available, set device to mps
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        super().__init__()
        self.hidden_size = hidden_size

        # 权重矩阵
        self.W_in = nn.Linear(input_size, hidden_size,device=device)
        self.W_rec = nn.Linear(hidden_size, hidden_size,device=device)

        # 初始时间常数（可训练）
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.5)

    def forward(self, x, h):
        # 计算输入驱动和递归连接
        x,h = x.to(self.device), h.to(self.device)
        u = self.W_in(x) + self.W_rec(h)

        # 动态微分方程近似更新
        dh = (-h + torch.tanh(u)) / self.tau
        h_new = h + dh  # Euler 步长默认 1

        return h_new

class LiquidRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.liquid_neuron = LiquidNeuron(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size,device=device)
        self.hidden_size = hidden_size

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        for t in range(x.size(1)):  # 遍历时间步
            h = self.liquid_neuron(x[:, t], h)

        out = self.fc(h)  # 用最后状态作预测
        return out
if __name__ == '__main__':

    # 构造随机序列（batch_size=32, seq_len=10, input_size=2）
    x = torch.randn(32, 10, 2)
    y = torch.randint(0, 2, (32,))
    x, y = x.to(device), y.to(device)
    model = LiquidRNN(input_size=2, hidden_size=16, output_size=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 简单训练循环
    for epoch in range(100):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
