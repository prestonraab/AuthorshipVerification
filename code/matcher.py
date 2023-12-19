import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from constants import *


class Matcher(Dataset):
    def __init__(self, author_folder: Path, test: bool):
        super().__init__()
        self.groups = defaultdict(list)
        files = list(author_folder.glob("*.pt"))
        for file in files:
            author = int(file.stem.split("_")[0])
            self.groups[author].append(file)
        authors = list(self.groups.keys())
        test_authors = random.sample(authors, int(len(authors) * 0.2))
        self.test_files = []
        self.train_files = []
        for author in authors:
            if author in test_authors:
                self.test_files += self.groups[author]
            else:
                self.train_files += self.groups[author]
        self.test = test

    def __len__(self):
        return len(self.test_files) * 10 if self.test else len(self.train_files)

    def __getitem__(self, index):
        file = random.choice(self.test_files) if self.test else self.train_files[index]
        author = int(file.stem.split("_")[0])
        group = self.groups[author]
        with open(file, 'rb') as f:
            first_vect = random.choice(torch.load(f))

        cls = (index % 2) * 2 - 1
        if cls == 1:
            other_file = random.choice(group)
        else:
            other_file = random.choice(self.test_files if self.test else self.train_files)
            while other_file in group:
                other_file = random.choice(self.test_files if self.test else self.train_files)

        with open(other_file, 'rb') as f:
            other_vect = random.choice(torch.load(f))
        return first_vect, other_vect, torch.tensor(cls, dtype=torch.float)


