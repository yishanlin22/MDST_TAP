import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset for CSV files
class MNISTCSVLoader(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)  # First column is label
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32) / 255.0  # Normalize pixel values
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load dataset from CSV
train_dataset = MNISTCSVLoader("sample_Data/mnist_train_small.csv")
test_dataset = MNISTCSVLoader("sample_Data/mnist_test.csv")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # TODO: Create the first fully connected layer (input: 28*28 pixels, output: 128 neurons)
        # TODO: Create the second fully connected layer (input: 128 neurons, output: 64 neurons)
        # TODO: Create the third (output) fully connected layer (input: 64 neurons, output: 10 classes)
        # TODO: Define the ReLU activation function

    def forward(self, x):
        # TODO: Apply the first fully connected layer and pass it through ReLU activation
        # TODO: Apply the second fully connected layer and pass it through ReLU activation
        # TODO: Apply the final output layer (do not use activation here)
        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        # TODO: Compute the loss using the criterion
        # TODO: Perform backpropagation to compute gradients
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Visualization of Top Features
def visualize_features(weight_matrix, title, num_features, grid_shape, reshape_size=None):
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    # Select the most "important" features based on absolute weight magnitude
    top_indices = np.argsort(np.abs(weight_matrix).sum(axis=1))[-num_features:]

    for i, idx in enumerate(top_indices):
        row, col = divmod(i, grid_shape[1])
        
        # Determine the reshape size dynamically
        feature_map = weight_matrix[idx]
        if reshape_size:
            feature_map = feature_map.reshape(reshape_size)

        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].axis('off')

    plt.show()

# Extract weights for visualization
fc1_weights = model.fc1.weight.data.cpu().numpy()  # (128, 784)
fc2_weights = model.fc2.weight.data.cpu().numpy()  # (64, 128)

# Visualize first hidden layer (Top 12 features)
visualize_features(fc1_weights, "Top 12 Features in First Hidden Layer", num_features=12, grid_shape=(3, 4), reshape_size=(28, 28))

# Visualize second hidden layer (Top 6 features) with a **16x8** reshape size
visualize_features(fc2_weights, "Top 6 Features in Second Hidden Layer", num_features=6, grid_shape=(2, 3), reshape_size=(16, 8))
# Evaluate the model and collect correctly classified examples
model.eval()
correct = 0
total = 0
correct_examples = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect correctly classified images
        for i in range(len(labels)):
            if predicted[i] == labels[i] and len(correct_examples) < 10:
                correct_examples.append((images[i].cpu().numpy().reshape(28, 28), labels[i].item(), predicted[i].item()))

# Print test accuracy
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot 10 correctly classified images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("10 Correctly Classified MNIST Images", fontsize=16)

for i, (image, true_label, predicted_label) in enumerate(correct_examples):
    row, col = divmod(i, 5)
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(f"True: {true_label}, Pred: {predicted_label}")
    axes[row, col].axis('off')

plt.show()
