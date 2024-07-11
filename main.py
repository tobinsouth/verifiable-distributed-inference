# Imports
import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import model_processing
import model_proving

# Constants
# Define constants
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
EPOCHS = 1
BATCH_SIZE = 64
MODEL_DIR = "./model"
DATA_DIR = "./data"
PROOF_DIR = "./proof"
MODEL_ID = "model_3"
VERBOSE = False

# Define Model, Train, Test, Shard functions
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
            if VERBOSE:
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
    if VERBOSE:
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    #
    # if VERBOSE:
    #     print("Starting Training")
    # for t in range(EPOCHS):
    #     if VERBOSE:
    #         print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # print("Finished Training")

    start = time.time()
    # Instantiate Model Processor
    model_processor = model_processing.Processor(
        model=model,
        sample_input=torch.randn(1, 1, 28, 28).to(DEVICE)
    )
    # Shard model
    # Comment out if you don't want to shard the model. All else stays equal.
    # model_processor.shard()


    # Save shards of model/entire model
    model_processor.save(
        model_id=MODEL_ID,
        storage_dir=MODEL_DIR
    )

    # checkpoint = time.time()

    model_prover = model_proving.Prover(
        model_id=MODEL_ID,
        model_dir=MODEL_DIR,
        proof_dir=PROOF_DIR,
        input_visibility="public",
        output_visibility="public",
        param_visibility="fixed",
        ezkl_optimization_goal="resources"
    )
    model_prover.old_prove()

    # end = time.time()
    print(f"Time taken: {end - start}")
