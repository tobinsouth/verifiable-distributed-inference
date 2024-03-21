# Imports
import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

# Constants
# Define constants
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
EPOCHS = 5
BATCH_SIZE = 64
MODEL_DIR = "./model"
DATA_DIR = "./data"
# Split prefix used in .onnx output files for each shard.
MODEL_SPLIT_PREFIX = "model_2"

# Define Model, Train, Test, Shard functions
# Define Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

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
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def shard(model):
    num_splits = len(list(model.children()))
    splits = []
    for i in range(num_splits):
        splits.append(nn.Sequential(list(model.children())[i]))
    return splits


if __name__ == "__main__":
    # Setup Training & Testing Data
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
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=(not MNIST_folder_exists),
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print("(Down)loaded MNIST dataset")

    # Instantiate Model
    model = Model().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print("Starting Training")
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Finished Training")

    # Split Model
    shards = shard(model)

    # Save shards
    sample_input_tensor = torch.randn(1, 1, 28, 28).to(DEVICE)
    start = time.time()
    for i in range(len(shards)):
        MODEL_FILE = f"{MODEL_SPLIT_PREFIX}_shard_{i}.onnx"
        model_shard = shards[i]
        torch.onnx.export(
            model=model_shard,
            args=sample_input_tensor,
            f=f"{MODEL_DIR}/{MODEL_FILE}",
            verbose=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"Saved {MODEL_FILE}")

        # Update sample input tensor for next layer/stack of layers
        model_shard.eval()
        sample_input_tensor = model_shard(sample_input_tensor)

    end = time.time()
    print(f"Time taken: {end - start}")
