import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
# 模拟数据
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10*np.pi, n_samples)
data = np.sin(t)
def generate_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    x = np.array(x)
    y = np.array(y)
    x = torch.from_numpy(x).unsqueeze(2).float()
    y = torch.from_numpy(y).unsqueeze(1).float()
    return x, y

# 分割数据集为训练集和测试集
train_ratio = 0.8
train_size = int(n_samples * train_ratio)
train_data = data[:train_size]
test_data = data[train_size:]

# 定义模型参数
input_size = 1#输入维度
hidden_size = 64
output_size = 1#输出维度
sequence_length = 10#每10个预测一次
num_layers = 2
learning_rate = 0.001#设置学习率
num_epochs = 100#训练多少个epoch

# 创建模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 创建模型实例
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
# 将训练数据转换为序列数据
x_train, y_train = generate_sequences(train_data, sequence_length)

# 开始训练
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

def show_origin():
    # 将测试数据转换为序列数据
    x_test, y_test = generate_sequences(test_data, sequence_length)

    # 关闭梯度计算
    with torch.no_grad():
        model.eval()
        predicted = model(x_test)

    # 将预测结果转换为numpy数组
    predicted = predicted.squeeze().numpy()

    # 绘制原始数据和预测结果
    plt.plot(test_data[sequence_length:], label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def show_more():
    # 获取最后一个输入序列
    last_sequence = data[-sequence_length:]
    last_sequence = torch.from_numpy(last_sequence).unsqueeze(0).unsqueeze(2).float()
    x_test, y_test = generate_sequences(test_data, sequence_length)
    # 关闭梯度计算
    with torch.no_grad():
        model.eval()
        predicted = model(x_test)
        model.eval()
        future_predictions = []#未来的预测结果存储的列表
        # 迭代进行未来5个时间步的预测
        for _ in range(5):
            # 预测下一个时间步
            next_prediction = model(last_sequence)
            future_predictions.append(next_prediction.item())
            # 更新输入序列，去掉第一个时间步，加入预测值作为最后一个时间步
            last_sequence = torch.cat([last_sequence[:, 1:, :], next_prediction.unsqueeze(0)], dim=1)


    # 将预测结果转换为numpy数组
    predicted = predicted.squeeze().numpy()
    future_predictions = np.array(future_predictions)
    # 组合预测结果
    all_predictions = np.concatenate((predicted, future_predictions))

    # 绘制原始数据和预测结果
    plt.plot(data, label='Actual')
    plt.plot(range(sequence_length, sequence_length + len(predicted)), predicted, label='Predicted')#测试集的数据进行预测的结果
    plt.plot(range(sequence_length + len(predicted), sequence_length + len(predicted) + len(future_predictions)),
             future_predictions, label='Future Predictions')#自己预测的结果
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
show_more()