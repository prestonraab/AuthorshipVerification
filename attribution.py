
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
    for author in author_vects:
        for vect in author_vects[author]:
            distances = [torch.cosine_similarity(vect.unsqueeze(0), average.unsqueeze(0)) for average in averages]
            average = sum(distances) / len(distances)
            standard_deviation = torch.std(torch.tensor(distances))
            average_distances_for_authors.append((average, standard_deviation))

    for index, author in enumerate(author_vects):
        if index % 2 == 0:
            print(index)
        for vect in author_vects[author]:
            total += 1
            similarities = [1 - torch.cosine_similarity(vect.unsqueeze(0), average.unsqueeze(0)) for average in averages]
            # subtract the average distance for each author
            similarities = [similarity - average_distances_for_authors[i][0] for i, similarity in enumerate(similarities)]
            # divide by the standard deviation for each author
            similarities = [abs(similarity / average_distances_for_authors[i][1]) for i, similarity in enumerate(similarities)]

            if similarities.index(min(similarities)) == index:
                correct += 1
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

        print(f"Author {author} has {len(author_vects[author])} vectors")
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

    author_folder = Path("datasets/authors")
    authors, author_files = get_files_for_authors(author_folder)

    siamese_network = Verifier().to(device)

    #load model
    siamese_network.load_state_dict(torch.load("siamese_network.pt"))


    N = 10

    correct = 0
    total = 0
    for epoch in range(10):
        print(f"Epoch {epoch}")
        n_random_authors = random.sample(authors, N)
        correct, total = accuracy_for_N_authors({author: author_files[author] for author in n_random_authors}, device, siamese_network)
        print(f"Accuracy for {N} authors: {correct / total:.4f}")
    print(f"Accuracy for {len(authors)} authors: {correct / total:.4f}")


if __name__ == "__main__":
    main()



