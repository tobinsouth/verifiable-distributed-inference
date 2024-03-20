# Imports
import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Constants
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
epochs = 5
batch_size = 64
MODEL_DIR = "./model"
DATA_DIR = "./data"
MODEL_FILE = "model.onnx"


# Define Model, Train and Test functions
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Setup Data
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # (Down)load training data
    MNIST_folder_exists = os.path.exists(f"{DATA_DIR}/MNIST")
    train_data = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=(not MNIST_folder_exists),
        transform=ToTensor()
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=(not MNIST_folder_exists),
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print("(Down)loaded MNIST dataset")

    # Run & Train Model
    # Instantiate Model
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print("Starting Training")
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Finished Training")

    # Save Model
    sample_input_tensor = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(model, sample_input_tensor, f"{MODEL_DIR}/{MODEL_FILE}", verbose=True, input_names=['input'],
                      output_names=['output'])
    print(f"Saved Model to {MODEL_DIR}/{MODEL_FILE}")
