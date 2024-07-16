# Imports
import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Constants
# Define constants
DEVICE = 'cpu'

EPOCHS = 1
BATCH_SIZE = 64
DATA_DIR = "./data"
PROOF_DIR = "./proof"
VERBOSE = False


# TODO: Find optimimal set of models (NEEDS to have 12 layers, so sharding works out nicely)

# Define Model
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(10, 10)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(10, 10)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        return x


# Update this with model definition above.
MODEL_DIMENSIONS = [
    (1, 1, 5, 5),
    (1, 25),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10),
    (1, 10)
]


class Trainer:

    def __init__(self,
                 load_training_data: bool = True):
        self.model = Model1().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.train_dataloader = None
        self.test_dataloader = None
        self.load_training_data = load_training_data
        if load_training_data:
            self.load_in_training_data()

    def load_in_training_data(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        # (Down)load training data
        mnist_folder_exists = os.path.exists(f"{DATA_DIR}/MNIST")
        train_data = datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=(not mnist_folder_exists),
            transform=ToTensor()
        )
        self.train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
        test_data = datasets.MNIST(
            root=DATA_DIR,
            train=False,
            download=(not mnist_folder_exists),
            transform=ToTensor()
        )
        self.test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
        if VERBOSE:
            print("(Down)loaded MNIST dataset")

    def train(self) -> None:
        if not self.load_training_data:
            print("Training data not loaded. Can't train model")
            return
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                if VERBOSE:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self) -> None:
        if not self.load_training_data:
            print("Test data not loaded. Can't train model")
            return
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if VERBOSE:
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    @staticmethod
    def get_dummy_input() -> torch.Tensor:
        # return torch.randn(1, 1, 28, 28).to(DEVICE)
        return torch.randn(1, 1, 5, 5).to(DEVICE)


