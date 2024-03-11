import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
PROOFS_DIR = "proofs"


# SimpleNet Model Definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # Using fewer filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 14 * 14, 10),  # Directly to output, no hidden layers
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


def split_model(model):
    # Split point: Example, after the convolutional layers
    model_part1 = nn.Sequential(*list(model.children())[:1])
    model_part2 = nn.Sequential(*list(model.children())[1:])
    return model_part1, model_part2


# MNIST Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the network and optimizer
net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.to(DEVICE)

# Train the network
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

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
net.to(DEVICE)

model_part1, model_part2 = split_model(net)

# Export the model to ONNX
os.makedirs(PROOFS_DIR, exist_ok=True)

dummy_input = torch.rand(inputs[0].unsqueeze(0).shape)  # Unbatched input
torch.onnx.export(model_part1, dummy_input, f"{PROOFS_DIR}/model_part1.onnx", verbose=True)

intermediate_output = model_part1(dummy_input)
intermediate_output = torch.flatten(intermediate_output, 1)

torch.onnx.export(model_part2, intermediate_output, f"{PROOFS_DIR}/model_part2.onnx", verbose=True)
