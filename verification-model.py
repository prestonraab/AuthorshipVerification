from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DistilBertModel

from constants import *
from verifier import Verifier
from matcher import Matcher
import swifter

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# How to make a siamese network
# https://github.com/pytorch/examples/blob/main/siamese_network/main.py

# The LFCC transform
# https://pytorch.org/audio/main/generated/torchaudio.transforms.LFCC.html

# How to build a model
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# My document
# https://docs.google.com/document/d/1DMt0IlimTDa8RaUJzh41m7DST-P0NdZ1HZibs5m6fPw/edit


def train(log_interval: int, model: Verifier, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.CosineEmbeddingLoss()

    for batch_idx, ((input_ids_1, attention_mask_1), (input_ids_2, attention_mask_2), targets) in enumerate(train_loader):
        input_ids_1, attention_mask_1 = input_ids_1.to(device), attention_mask_1.to(device)
        input_ids_2, attention_mask_2 = input_ids_2.to(device), attention_mask_2.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model({'input_ids': input_ids_1, 'attention_mask': attention_mask_1}, {'input_ids': input_ids_2, 'attention_mask': attention_mask_2})
        output1, output2 = output
        loss = criterion(output1, output2, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_ids_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0

    criterion = nn.CosineEmbeddingLoss()

    with torch.no_grad():
        for ((input_ids_1, attention_mask_1), (input_ids_2, attention_mask_2), targets) in test_loader:
            input_ids_1, attention_mask_1 = input_ids_1.to(device), attention_mask_1.to(device)
            input_ids_2, attention_mask_2 = input_ids_2.to(device), attention_mask_2.to(device)
            targets = targets.to(device)
            output = model({'input_ids': input_ids_1, 'attention_mask': attention_mask_1}, {'input_ids': input_ids_2, 'attention_mask': attention_mask_2})
            output1, output2 = output
            print(f"\tTarget: {targets}")
            test_loss += criterion(output1, output2, targets).sum().item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def report_time(start_time):
    print(f"Time elapsed: {time.time() - start_time:.4f}")
    return time.time()

def main():
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

    s = time.time()

    if LOAD_DATASETS_FROM_FILE:
        print("Loading training set...")
        train_dataset = torch.load("train_dataset.pt")
        s = report_time(s)
        print("Loading test set...")
        test_dataset = torch.load("test_dataset.pt")
        s = report_time(s)
    else:
        # get blogs dataset
        print("Reading CSV...")
        blogs = pd.read_csv(Path("datasets/blog8965.csv.gz")).dropna()
        s = report_time(s)
        print("Filtering blogs...")
        blogs = blogs[blogs["text"].swifter.apply(str.split).swifter.apply(len) > 38]
        s = report_time(s)

        # Load the dataset
        print("Loading training set...")
        train_dataset = Matcher(blogs, train=True, tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"))
        s = report_time(s)

        print("Saving training dataset...")
        torch.save(train_dataset, "train_dataset.pt")
        s = report_time(s)

        print("Loading test set...")
        test_dataset = Matcher(blogs, train=False, tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"))
        s = report_time(s)

        print("Saving train dataset...")
        torch.save(test_dataset, "test_dataset.pt")
        s = report_time(s)

    # Create the dataloaders
    print("Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    s = report_time(s)

    verifier = Verifier().to(device)

    if LOAD_MODEL:
        print("Loading model...")
        verifier.load_state_dict(torch.load("siamese_network.pt"))
        s = report_time(s)

    bert = verifier.bert
    for param in bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(verifier.parameters())
    # optimizer = torch.optim.Adadelta(verifier.parameters(), lr=LEARNING_RATE)

    #scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    print("Beginning training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print("Epoch " + str(epoch))
        train(LOG_INTERVAL, verifier, device, train_loader, optimizer, epoch)
        test(verifier, device, test_loader)
        # scheduler.step()
        s = report_time(s)

        if SAVE_MODEL:
            torch.save(verifier.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()
