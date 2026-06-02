import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.ao.quantization import quantize_dynamic
import os
import time
import matplotlib.pyplot as plt


# Define the CNN model:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input channels=1, output channels=32, kernel size=3x3, stride=1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second convolutional layer: input channels=32, output channels=64, kernel size=3x3, stride=1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Third convolutional layer: input channels=64, output channels=128, kernel size=3x3, stride=1
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        # First fully connected layer: input features=5*5*128, output features=512
        self.fc1 = nn.Linear(5*5*128, 512)
        # Second fully connected layer: input features=512, output features=10 (number of classes)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        # Apply first convolutional layer and ReLU activation
        x = torch.relu(self.conv1(x))
        # Apply second convolutional layer and ReLU activation
        x = torch.relu(self.conv2(x))
        # Apply max pooling with kernel size=2x2
        x = torch.max_pool2d(x, 2)
        # Apply third convolutional layer and ReLU activation
        x = torch.relu(self.conv3(x))
        # Apply another max pooling with kernel size=2x2
        x = torch.max_pool2d(x, 2)
        # Flatten the tensor starting from dimension 1
        x = torch.flatten(x, 1)
        # Apply first fully connected layer and ReLU activation
        x = torch.relu(self.fc1(x))
        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
        return x


# Helper function to train the model for one epoch
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data and targets to the specified device
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass: compute predicted outputs
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Update model parameters
        if batch_idx % 100 == 0:
            # Print training status every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0  # Initialize cumulative loss
    correct = 0  # Initialize count of correct predictions
    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data and targets to the specified device
            output = model(data)  # Forward pass: compute predicted outputs
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    test_loss /= len(test_loader.dataset)  # Average loss over all test samples
    accuracy = 100. * correct / len(test_loader.dataset)  # Calculate accuracy percentage
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return test_loss, accuracy



# Set up data loaders with appropriate transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean=0.1307 and std=0.3081
])

# Download and load the training dataset (MNIST)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# Download and load the test dataset (MNIST)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
# Create data loaders for training and testing with specified batch sizes
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Training the non quantized model:

# Device configuration: use GPU if available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the CNN model and move it to the selected device
non_quant_model = CNN().to(device)

# Define the optimizer (Adam) with model parameters
optimizer = optim.Adam(non_quant_model.parameters())

# Define the loss function (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Train and evaluate the non-quantized model for one epoch (for demonstration purposes)
print("Training and evaluating the non-quantized model...")
for epoch in range(1):  # Loop over the dataset once
    train(non_quant_model, device, train_loader, optimizer, criterion, epoch)  # Train the model
non_quant_loss, non_quant_accuracy = test(non_quant_model, device, test_loader)



# Save the non-quantized model's state dictionary to a file
torch.save(non_quant_model.state_dict(), "non_quant_model.pth")
# Get the size of the saved model file in megabytes
non_quant_size = os.path.getsize("non_quant_model.pth") / (1024 * 1024)  # Size in MB

def measure_inference_time(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    start_time = time.time()  # Record the start time
    with torch.no_grad():  # Disable gradient calculation
        for data, _ in test_loader:
            data = data.to(device)  # Move data to the specified device
            _ = model(data)  # Perform a forward pass (ignore the output)
    end_time = time.time()  # Record the end time
    return end_time - start_time  # Return the total inference time
# Measure inference time for the non-quantized model
non_quant_inference_time = measure_inference_time(non_quant_model, device, test_loader)


# Quantize the model using dynamic quantization provided by torch.ao
print("\nQuantizing the model using torch.ao...")
quant_model = quantize_dynamic(
    non_quant_model.cpu(),  # Move the model to CPU before quantization
    {nn.Linear, nn.Conv2d},  # Specify layers to quantize (Linear and Conv2d)
    dtype=torch.qint8  # Set the quantization data type to 8-bit integers
)


# Test the quantized model on CPU
quant_loss, quant_accuracy = test(quant_model, torch.device('cpu'), test_loader)


# Save the quantized model's state dictionary to a file
torch.save(quant_model.state_dict(), "quant_model.pth")
# Get the size of the saved quantized model file in megabytes
quant_size = os.path.getsize("quant_model.pth") / (1024 * 1024)  # Size in MB# Measure inference time for the quantized model
quant_inference_time = measure_inference_time(quant_model, torch.device('cpu'), test_loader)


# Print the comparison results between non-quantized and quantized models
print("\nResults:")
print(f"Non-quantized model size: {non_quant_size:.2f} MB")
print(f"Quantized model size: {quant_size:.2f} MB")
print(f"Non-quantized model accuracy: {non_quant_accuracy:.2f}%")
print(f"Quantized model accuracy: {quant_accuracy:.2f}%")
print(f"Non-quantized inference time: {non_quant_inference_time:.4f} seconds")
print(f"Quantized inference time: {quant_inference_time:.4f} seconds")

# Visualization of the results using a bar chart
labels = ['Model Size (MB)', 'Accuracy (%)', 'Inference Time (s)']
non_quant_values = [non_quant_size, non_quant_accuracy, non_quant_inference_time]
quant_values = [quant_size, quant_accuracy, quant_inference_time]
x = range(len(labels))  # Position of the bars on the x-axis
width = 0.35  # Width of each bar
fig, ax = plt.subplots(figsize=(12, 6))  # Create a figure and a set of subplots
# Plot non-quantized model metrics
ax.bar([i - width/2 for i in x], non_quant_values, width, label='Non-quantized')
# Plot quantized model metrics
ax.bar([i + width/2 for i in x], quant_values, width, label='Quantized')
ax.set_ylabel('Values')  # Set the label for the y-axis
ax.set_title('Comparison of Non-quantized vs Quantized Model')  # Set the title of the plot
ax.set_xticks(x)  # Set the positions of the x-ticks
ax.set_xticklabels(labels)  # Set the labels for the x-ticks
ax.legend()  # Add a legend to differentiate the bars
plt.tight_layout()  # Adjust the padding between and around subplots
plt.show()  # Display the plot