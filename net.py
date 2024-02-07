## 炼丹炉
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data.csv'
data = pd.read_csv(file_path)
X = data['初试成绩'].values.reshape(-1, 1)
y = data['复试成绩'].values.reshape(-1, 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=0)

X_train_torch = torch.tensor(X_train.astype(np.float32))
y_train_torch = torch.tensor(y_train.astype(np.float32))
X_test_torch = torch.tensor(X_test.astype(np.float32))
y_test_torch = torch.tensor(y_test.astype(np.float32))

train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 500)
        self.fc2 = nn.Linear(500, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 30000
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
net.eval()
with torch.no_grad():
    predictions = net(X_test_torch)

predictions_np = predictions.numpy()

test_set_sorted = pd.DataFrame({'X_test': X_test.reshape(-1), 'y_test': y_test.reshape(-1), 'Predictions': predictions_np.reshape(-1)})
test_set_sorted.sort_values(by='X_test', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(test_set_sorted['X_test'], test_set_sorted['y_test'], label='Actual', marker='o')
plt.plot(test_set_sorted['X_test'], test_set_sorted['Predictions'], label='Predicted', marker='x')
plt.title('Actual vs Predicted Scores - Neural Network')
plt.xlabel('Scaled Initial Test Score')
plt.ylabel('Resit Test Score')
plt.legend()
plt.grid(True)
plt.show()
