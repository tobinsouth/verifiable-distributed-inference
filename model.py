
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Training Parameters
batch_size = 64
epochs = 5
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


# Smaller CNN Model
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 2 * 2, 128),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# MNIST Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize the network and optimizer
net = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.to(device)

# Train the network
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# Now we split up the model

net.to("cpu")

def split_model(model):
    # Split point: Example, after the convolutional layers
    model_part1 = nn.Sequential(*list(model.children())[:1])  
    model_part2 = nn.Sequential(*list(model.children())[1:]) 
    return model_part1, model_part2

model_part1, model_part2 = split_model(net)


# Export the model to ONNX    
os.makedirs("proofs", exist_ok=True)

dummy_input = torch.rand(inputs[0].unsqueeze(0).shape) # Unbatched input
torch.onnx.export(model_part1, dummy_input, "proofs/model_part1.onnx", verbose=True)

intermediate_output = model_part1(dummy_input)
intermediate_output = torch.flatten(intermediate_output, 1)

torch.onnx.export(model_part2, intermediate_output, "proofs/model_part2.onnx", verbose=True)
