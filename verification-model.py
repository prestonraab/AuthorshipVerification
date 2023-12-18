from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import *
from verifier import Verifier
from matcher import Matcher

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

class SimplePerceptron(nn.Module):
    def __init__(self):
        super(SimplePerceptron, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.linear.weight, 0.001)
        self.linear.bias.data.fill_(0.001)

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


def train(model: Verifier, simple_model, device, train_loader, optimizer, optimizer2, epoch, s):
    model.train()
    simple_model.train()

    criterion = nn.CosineEmbeddingLoss(reduction="mean")
    criterion2 = nn.BCELoss()

    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    start = time.time()
    for batch_idx, (x1, x2, targets) in enumerate(train_loader):
        x1, x2 = x1.to(device), x2.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        optimizer2.zero_grad()
        output1, output2 = model(x1, x2)
        loss = criterion(output1, output2, targets)

        cos = cosine(output1, output2)

        simple_output = simple_model(cos.unsqueeze(1))

        loss2 = criterion2(simple_output, ((targets + 1.0) / 2.0).unsqueeze(1))

        loss.backward(retain_graph=True)
        loss2.backward()

        optimizer.step()
        optimizer2.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(targets):5.0f}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]"
                  f"\tLoss: {loss.item():.4f}"
                  f"\tLoss 2: {loss2.item():.4f}"
                  f"\tSimple_params: {list(simple_model.parameters())[0].item():.4f}"
                  f"\t{'#' * int(100 * loss.item())}")
    print("########################\n" * 3)
    print(f"Finished in {time.time() - start:.4f} seconds")
    print("########################\n" * 3)


def test(model, simple_model, device, test_loader: DataLoader):
    model.eval()
    simple_model.eval()
    test_loss = 0
    num_correct = 0
    num_total = 0

    criterion = nn.CosineEmbeddingLoss()
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
    i = 0
    with torch.no_grad():
        for x1, x2, targets in test_loader:
            num_total += len(targets)
            x1, x2 = x1.to(device), x2.to(device)
            targets = targets.to(device)
            output = model(x1, x2)
            output1, output2 = output
            first_0_target = None
            for j, target in enumerate(targets):
                if target == -1:
                    first_0_target = j
                    break
            print(f"\tTarget: {targets}")
            test_loss += criterion(output1, output2, targets).item()

            cos = cosine(output1, output2)

            simple_output = simple_model(cos.unsqueeze(1))
            num_correct += torch.sum(((simple_output > 0.5) == ((targets + 1.0) // 2 ).unsqueeze(1)).float()).item()

            print(f"\nOutput 1:\n{output1[first_0_target]}")
            print(f"\nOutput 2:\n{output2[first_0_target]}")
            zero_loss = criterion(output1[first_0_target], output2[first_0_target], torch.tensor(-1).to(device))
            print(f"\tLoss: {test_loss}")
            print(f"\tZero loss: {zero_loss}")
            i += 1
    test_loss /= i

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    print(f"Accuracy: {num_correct / num_total:.4f}")


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

    author_folder = Path("datasets/authors")
    train_matcher = Matcher(author_folder, False)
    test_matcher = Matcher(author_folder, True)
    train_loader = DataLoader(train_matcher, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_matcher, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    s = report_time(s)

    verifier = Verifier().to(device)
    simple_model = SimplePerceptron().to(device)

    if LOAD_MODEL:
        print("Loading model...")
        verifier.load_state_dict(torch.load("siamese_network_trained_2.pt"))
        s = report_time(s)

    try:
        simple_model.load_state_dict(torch.load("simple_model.pt"))
        print("Loading simple model...")
        s = report_time(s)
    except:
        print("Unable to load simple model.")
        pass

    optimizer = torch.optim.AdamW(verifier.parameters())
    optimizer2 = torch.optim.AdamW(simple_model.parameters())
    # optimizer = torch.optim.Adadelta(verifier.parameters(), lr=LEARNING_RATE)

    print("Testing model...")
    test(verifier, simple_model, device, test_loader)

    # scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    print("Beginning training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print("Epoch " + str(epoch))
        train(verifier, simple_model, device, train_loader, optimizer, optimizer2, epoch, s)
        # scheduler.step()
        s = report_time(s)

        if DO_TESTING and epoch % TEST_INTERVAL == 0:
            print("Testing model...")
            test(verifier, simple_model, device, test_loader)
            s = report_time(s)

        if epoch % SAVE_INTERVAL == 0:
            print("Saving model...")
            torch.save(verifier.state_dict(), "siamese_network.pt")
            torch.save(simple_model.state_dict(), "simple_model.pt")
            s = report_time(s)

    torch.save(verifier.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()
