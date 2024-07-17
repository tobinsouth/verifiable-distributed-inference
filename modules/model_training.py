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


# Define Models
class LinearReluModel(nn.Module):
    name = 'linear_relu'

    model_dimensions = [
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


class CNNModel(nn.Module):
    name = 'cnn'

    model_dimensions = [
        (32, 1, 28, 28),
        (32, 10, 28, 28),
        (32, 10, 28, 28),
        (32, 10, 14, 14),
        (32, 25, 14, 14),
        (32, 25, 14, 14),
        (32, 25, 7, 7),
        (32, 1225),
        (32, 50),
        (32, 50),
        (32, 10),
        (32, 10)
    ]

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, (5, 5), padding=2)  # Adding padding to maintain spatial dimensions
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(10, 25, (5, 5), padding=2)  # Adding padding to maintain spatial dimensions
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(25 * 7 * 7, 50)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(50, 10)
        self.relu4 = nn.ReLU()

        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)  # Flatten the tensor

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        x = self.fc4(x)
        return x


class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attention_output, _ = self.self_attention(x, x, x)
        x = x + attention_output
        x = self.norm(x)
        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output
        return x


class AttentionModel(nn.Module):
    name = 'attention'

    model_dimensions = [
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128),
        (10, 32, 128)
    ]

    num_heads_per_layer = 4

    def __init__(self):
        super().__init__()
        self.attention1 = AttentionLayer(self.model_dimensions[0][2], self.num_heads_per_layer)
        self.attention2 = AttentionLayer(self.model_dimensions[1][2], self.num_heads_per_layer)
        self.attention3 = AttentionLayer(self.model_dimensions[2][2], self.num_heads_per_layer)
        self.attention4 = AttentionLayer(self.model_dimensions[3][2], self.num_heads_per_layer)
        self.attention5 = AttentionLayer(self.model_dimensions[4][2], self.num_heads_per_layer)
        self.attention6 = AttentionLayer(self.model_dimensions[5][2], self.num_heads_per_layer)
        self.attention7 = AttentionLayer(self.model_dimensions[6][2], self.num_heads_per_layer)
        self.attention8 = AttentionLayer(self.model_dimensions[7][2], self.num_heads_per_layer)
        self.attention9 = AttentionLayer(self.model_dimensions[8][2], self.num_heads_per_layer)
        self.attention10 = AttentionLayer(self.model_dimensions[9][2], self.num_heads_per_layer)
        self.attention11 = AttentionLayer(self.model_dimensions[10][2], self.num_heads_per_layer)
        self.attention12 = AttentionLayer(self.model_dimensions[11][2], self.num_heads_per_layer)

    def forward(self, x):
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        x = self.attention4(x)
        x = self.attention5(x)
        x = self.attention6(x)
        x = self.attention7(x)
        x = self.attention8(x)
        x = self.attention9(x)
        x = self.attention10(x)
        x = self.attention11(x)
        x = self.attention12(x)
        return x


AVAILABLE_MODELS = [
    LinearReluModel.name,
    CNNModel.name,
    AttentionModel.name
]


class Trainer:

    def __init__(self,
                 load_training_data: bool = True,
                 model_name: str = ""):
        # check if there's a specific model that should be used
        if model_name == LinearReluModel.name:
            self.model = LinearReluModel().to(DEVICE)
        elif model_name == CNNModel.name:
            self.model = CNNModel().to(DEVICE)
        elif model_name == AttentionModel.name:
            self.model = AttentionModel().to(DEVICE)
        else:
            print(f'Invalid model name provided (options: {", ".join(AVAILABLE_MODELS)}), '
                  f'reverting to {LinearReluModel.name}')
            self.model = LinearReluModel().to(DEVICE)
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
