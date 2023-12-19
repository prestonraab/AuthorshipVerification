
import random
from collections import defaultdict

import torch
from pathlib import Path

from constants import FINAL_VECTOR_SIZE
from verifier import Verifier


def average_similarity(vect, author_vects):
    # Find the similarity between vect and each vector in author_vects
    # Return the average
    similarities = torch.cosine_similarity(vect.unsqueeze(0), author_vects)
    return similarities.mean()

def accuracy_for_N_authors(author_files, device, siamese_network):
    author_vects = get_author_vects(author_files, device, siamese_network)
    correct = 0
    total = 0
    averages = []
    for author in author_vects:
        averages.append(author_vects[author].mean(0))

    average_distances_for_authors = []
    for index, author in enumerate(author_vects):
        distances = [torch.cosine_similarity(vect.unsqueeze(0), averages[index].unsqueeze(0)) ** 2 for vect in author_vects[author]]
        average = (sum(distances) ** 0.5) / len(distances)
        average_distances_for_authors.append(average)

    print(average_distances_for_authors)

    for index, author in enumerate(author_vects):
        if index % (len(author_vects)//10) == 0:
            print(index)
        for vect in author_vects[author]:
            total += 1
            similarities = [torch.cosine_similarity(vect.unsqueeze(0), average.unsqueeze(0)) for average in averages]
            # divide by average distance for author
            similarities = [similarity / average_distances_for_authors[i][0] for i, similarity in enumerate(similarities)]

            if similarities.index(max(similarities)) == index:
                correct += 1
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    return correct, total


# For each author, find all of their documents
# For each document, create a vector, using siamese_network.pt
# For efficiency, do two at a time
# Save each vector to a file.
def get_files_for_authors(author_folder):
    author_files= defaultdict(list)
    for file in author_folder.iterdir():
        print(file)
        try:
            author = int(file.stem.split("_")[0])
        except:
            pass
        author_files[author].append(file)
    authors = list(author_files.keys())
    print(f"Found {len(authors)} authors")
    return authors, author_files



def get_author_vects(author_files, device, siamese_network):
    author_vects = {}
    for author in author_files:
        author_vects[author] = torch.empty(0, FINAL_VECTOR_SIZE).to(device)
        # pick two at a time
        for file in author_files[author]:
            with open(file, 'rb') as f:
                fft = torch.load(f).to(device)
            with torch.no_grad():
                output1, output2 = siamese_network(fft.unsqueeze(0).to(device), fft.unsqueeze(0).to(device))
                author_vects[author] = torch.cat((author_vects[author], output1.squeeze(0)))

    return author_vects


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    author_folder = Path("/Users/prestonraab/GitHub/Ling/AuthorshipVerification/code/datasets/authors")
    authors, author_files = get_files_for_authors(author_folder)

    siamese_network = Verifier().to(device)
    siamese_network.eval()

    #load model
    siamese_network.load_state_dict(torch.load("/Users/prestonraab/GitHub/Ling/AuthorshipVerification/code/siamese_network.pt"))


    N = 10

    correct = 0
    total = 0
    accuracies = []
    for epoch in range(100):
        print(f"Epoch {epoch}")
        n_random_authors = random.sample(authors, N)
        correct, total = accuracy_for_N_authors({author: author_files[author] for author in n_random_authors}, device, siamese_network)
        print(f"Accuracy for {N} authors: {correct / total:.4f}")
        accuracies.append(correct / total)
        print(f"Accuracy so far: {sum(accuracies) / len(accuracies):.4f}")
        standard_deviation = torch.std(torch.tensor(accuracies))
        print(f"Standard deviation: {standard_deviation:.4f}")
    print(f"Accuracy for {len(authors)} authors: {correct / total:.4f}")


if __name__ == "__main__":
    main()



