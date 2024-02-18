import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
## 图像显示中文的问题
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False

def predict(data, train_window, future_pred, label, epochs,lr):
    # 转换为 numpy 数组
    data_np = np.array(data[label]).reshape(-1, 1)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data_np)

    # 转换为 PyTorch tensors
    data_normalized = torch.FloatTensor(data_normalized).view(-1)

    # 创建数据集
    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(data_normalized, train_window)

    # 定义 LSTM 模型
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    # 实例化模型
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # 训练模型
    # epochs = 25

    for epoch in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 25 == 1:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {epoch:3} loss: {single_loss.item():10.10f}')

    # 使用模型对整个数据集进行预测以创建拟合与真实数据对比图
    # 为预测做准备
    test_inputs = data_normalized[-train_window:].tolist()

    # 预测未来
    model.eval()
    for i in range(future_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    # 将预测数据转换回原始规模
    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

    # 输出预测的未来私营企业乡村就业人员数值
    predicted_years = np.arange(data["年份"][-1] + 1 - train_window, data["年份"][-1] + 1 + future_pred)
    for year, population in zip(predicted_years, actual_predictions):
        print(f"年份: {year}, 预测{label}: {population[0]:.2f}")

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.title(f'未来{label}变化预测')
    plt.xlabel('年份')
    plt.ylabel(f'{label}')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    # 绘制实际数据
    plt.plot(data["年份"], data[label], label='Real Data')
    
    # 绘制预测数据
    plt.plot(predicted_years, np.concatenate([data[label][-train_window:], actual_predictions.ravel()]), label='Predictions', linestyle='--')
    
    plt.legend()
    plt.show()
    
data1 = {
    "年份": [2021,2022,2023],
    "杭电081200均分": [356,332,358]
}

predict(data1, train_window=2, future_pred=3, label="杭电081200均分",epochs=50,lr=0.005)