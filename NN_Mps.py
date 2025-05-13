import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# Generate synthetic data for classification
X = torch.randn(100, 100)  # 10,000 samples, 100 features
y = (torch.rand(100) > 0.5).long()  # Binary classification labels (0 or 1)

gpu_results = []
cpu_results = []
output = []

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run_on_cpu(X, y, epoch):
    # Setup for CPU training
    device_cpu = torch.device("cpu")
    model_cpu = SimpleNN().to(device_cpu)
    X_cpu = X.to(device_cpu)
    y_cpu = y.to(device_cpu)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cpu.parameters(), lr=0.001)

    # Benchmark CPU training time
    start_cpu = time.time()
    for epoch in range(epoch):
        optimizer.zero_grad()
        outputs = model_cpu(X_cpu)
        loss = criterion(outputs, y_cpu)
        loss.backward()
        optimizer.step()
    end_cpu = time.time()
    cpu_results.append(end_cpu - start_cpu)
    print("CPU Training time: {:.2f} seconds".format(end_cpu - start_cpu))
    return end_cpu - start_cpu



def run_on_gpu(X, y, epoch):
    # Check if MPS is available and set the device accordingly
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.........")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU for training.")

    # Move model and data to the selected device
    model = SimpleNN().to(device)
    X = X.to(device)
    y = y.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Benchmark MPS training time
    start_time = time.time()
    for epoch in range(epoch):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # For MPS, synchronize to ensure all operations are completed
    if device.type == "mps":
        torch.mps.synchronize()
    end_time = time.time()
    gpu_results.append(end_time - start_time)
    print("GPU Training time: {:.2f} seconds\n".format(end_time - start_time))
    return end_time - start_time


epochs = []

for epoch in [10, 100]:
    epochs.append(epoch)
    output.append([epoch, run_on_cpu(X, y, epoch)])
    output.append([epoch, run_on_gpu(X, y, epoch)])

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, gpu_results, marker='o', linestyle='-', label='GPU', color='blue')
plt.plot(epochs, cpu_results, marker='o', linestyle='-', label='CPU', color='orange')
plt.xlabel('Number of Epochs')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Epochs (Two Runs)')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("output.png")