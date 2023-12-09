from collections import defaultdict

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import sys

# import Perceptron
from constants import *
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import swifter
import os
import time
import random
from multiprocessing import Pool
from torch.utils.data import Dataset

class SingleLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerPerceptron, self).__init__()
        # learnable weights and bias
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x1: Tensor, x2: Tensor):
        x = torch.cat((x1, x2), dim=1)
        y = torch.matmul(x, self.weight) + self.bias
        return torch.sigmoid(y)

    def predict(self, x1, x2):
        with torch.no_grad():
            return self.forward(x1, x2).round()

    def accuracy(self, x1, x2, y):
        return self.predict(x1, x2).eq(y).float().mean()


class Matcher(Dataset):
    def __init__(self, author_folder: Path):
        super().__init__()
        self.groups = defaultdict(list)
        self.files = list(author_folder.glob("*.pt"))
        for file in self.files:
            author = int(file.stem.split("_")[0])
            self.groups[author].append(file)

    def __len__(self):
        return 200

    def __getitem__(self, index):
        file = random.choice(self.files)
        author = int(file.stem.split("_")[0])
        group = self.groups[author]
        file = torch.load(file)
        file = random.choice(file)
        cls = index % 2
        if cls == 1:
            other_file = random.choice(group)
        else:
            other_file = random.choice(self.files)
            while other_file in group:
                other_file = random.choice(self.files)

        other_file = torch.load(other_file)
        other_file = random.choice(other_file)
        return file, other_file, torch.tensor([cls], dtype=torch.float32)


# Need to:
# train 768 Single layer perceptrons, one for each output of bert
# Make a DataSet that will load files
# Files should be randomly selected to be from the same author 1/2 the time
# Need a dictionary of author numbers to filenames
# Need to get all author numbers by examining folder
# Folder is found at /datasets/authors

# Need to sort the perceptrons by accuracy.

def train_perceptrons(perceptrons, train_loader, device):
    criterion = nn.BCELoss()
    print(f"Training")
    for i in range(3):
        print(f"train-Epoch {i}")
        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            for index, perceptron in enumerate(perceptrons):
                output = perceptron(x1[:,index::768], x2[:,index::768])
                loss = criterion(output, y)
                loss.backward()


def find_accuracies(perceptrons, train_loader, device):
    print("Testing")
    accuracies = []
    x1, x2, y = next(iter(train_loader))
    x1 = x1.to(device)
    x2 = x2.to(device)
    y = y.to(device)
    for i, perceptron in enumerate(perceptrons):
        accuracy = perceptron.accuracy(x1[:, i::768], x2[:, i::768], y).cpu()
        accuracies.append(accuracy)
    return accuracies


def find_features():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # The following lines are required to enable the mps backend
    if device == "mps":
        torch.multiprocessing.set_start_method("spawn", force=True)

    perceptrons = []
    for i in range(768):
        perceptron = SingleLayerPerceptron(514, 1).to(device)
        perceptrons.append(perceptron)

    # load perceptron if file exists
    for i in range(768):
        if os.path.exists(f"models/perceptron_{i}.pt"):
            perceptrons[i].load_state_dict(torch.load(f"models/perceptron_{i}.pt"))

    author_folder = Path("datasets/authors")
    matcher = Matcher(author_folder)
    train_loader = DataLoader(matcher, batch_size=200, shuffle=True, num_workers=3)
    test_loader = DataLoader(matcher, batch_size=200, shuffle=True, num_workers=3)

    best20 = set()
    epoch = 0
    perceptron_stats = {i: 0 for i in range(768)}
    while True:
        epoch += 1
        print(f"Epoch {epoch}")
        train_perceptrons(perceptrons, train_loader, device)
        accuracies = find_accuracies(perceptrons, test_loader, device)

        best_20_indices = np.argsort(accuracies)[-20:]
        best_20_acc = np.array(accuracies)[best_20_indices]

        new_best20 = set(best_20_indices)
        intersection = new_best20.intersection(best20)
        print(f"Intersection: {intersection}")
        for i in intersection:
            perceptron_stats[i] += 1
            torch.save(perceptrons[i].state_dict(), f"models/perceptron_{i}.pt")
        if len(intersection) == 20:
            break
        best20 = new_best20

        true_best_20 = sorted(perceptron_stats, key=perceptron_stats.get, reverse=True)[0:20]
        table = []
        for p, acc, in zip(true_best_20, best_20_acc):
            print(f"Perceptron {p} has accuracy {acc:.3f}, has been best {perceptron_stats[p]} times")
            table.append([p, acc, perceptron_stats[p]])
        pd.DataFrame(table).to_csv("perceptron_stats.csv", index=False)


if __name__ == '__main__':
    find_features()






