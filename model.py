import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import categorize_street, encode_street_type, encode_traffic_signal, encode_crossing

print("Loading US Accidents dataset...")

file_path = 'data/US_Accidents_March23.csv'

# Use the skiprows parameter to load only the desired row
step = 77
# Generate a list of rows to skip
rows_to_skip = [i for i in range(1, 7728394) if i % step != 0]  # Skip all except every 77th row

# Read the CSV with the selected rows
us_accidents = pd.read_csv(file_path, skiprows=rows_to_skip)

# Verify the size of the sample
print(f"Sample size: {len(us_accidents)}")

print("Preprocessing the dataset...")

# Select target + features with predictive power
selected_columns = ['Severity', 'Traffic_Signal', 'Crossing', 'Street', 'Distance(mi)']
us_accidents = us_accidents[selected_columns]

# Drop rows with missing values in selected columns
us_accidents = us_accidents.dropna(subset=selected_columns)

# Encode categorical variables

us_accidents['Street_Type'] = us_accidents['Street'].apply(categorize_street)
us_accidents['Highway_Flag'] = us_accidents['Street_Type'].apply(encode_street_type)
us_accidents['Traffic_Signal_Flag'] = us_accidents['Traffic_Signal'].apply(encode_traffic_signal)
us_accidents['Crossing_Flag'] = us_accidents['Crossing'].apply(encode_crossing)
us_accidents.drop(columns=['Street', 'Street_Type', 'Traffic_Signal', 'Crossing'], inplace=True)

# Display the first few rows of the dataset
print("First few rows of the dataset: ")
print(us_accidents.head())

# Split the dataset into features and target
X = us_accidents[['Traffic_Signal_Flag', 'Crossing_Flag', 'Highway_Flag', 
                  'Distance(mi)']].values
y = us_accidents['Severity'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y - 1, dtype=torch.long)  # Shift labels from 1-4 to 0-3

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        # TODO: Create the first fully connected layer (input: 4 features, output: 128 neurons)
        
        # TODO: Create the second fully connected layer (input: 128 neurons, output: 64 neurons)

        # TODO: Create the third fully connected layer (input: 64 neurons, output: 32 neurons)
        
        # TODO: Create the fourth (output) fully connected layer (input: 32 neurons, output: 4 classes)
        
        # TODO: Define the ReLU activation function
        

    def forward(self, x):
        # TODO: Apply the first fully connected layer and pass it through ReLU activation
        
        # TODO: Apply the second fully connected layer and pass it through ReLU activation

        # TODO: Apply the third fully connected layer and pass it through ReLU activation
        
        # TODO: Apply the final output layer (do not use activation here)
        
        # return x
        return x

# Initialize model, loss function, and optimizer
model = AccidentSeverityModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_targets = []

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Compute predictions
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(targets.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Compute train accuracy
    train_accuracy = accuracy_score(all_train_targets, all_train_preds)
    train_accuracies.append(train_accuracy)

    # Evaluate model on test set
    model.eval()
    total_test_loss = 0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

            # Compute predictions
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    # Compute test accuracy
    test_accuracy = accuracy_score(all_test_targets, all_test_preds)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Plot training vs test loss
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()