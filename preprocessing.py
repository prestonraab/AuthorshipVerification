import numpy as np
import pandas as pd
from pathlib import Path
import swifter
import os
CHUNK_SIZE=1000
from transformers import AutoTokenizer
from constants import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from multiprocessing import Pool

from transformers import AutoTokenizer, DistilBertModel

from constants import *
import swifter
from torchaudio.transforms import Spectrogram


def random_truncate(text):
    split = text.split()
    length = len(split)
    if length >= 512:
        start = np.random.randint(length - 511)
        text = ' '.join(split[start:start + 511])
    return text


def main():
    blogs = pd.read_csv(Path("datasets/blog8965.csv.gz")).dropna()
    blogs = blogs[blogs["text"].swifter.apply(str.split).swifter.apply(len) > 38]

    print("\tTruncating...")
    blogs["text"] = blogs["text"].swifter.apply(random_truncate)

    dataset = blogs.to_numpy()
    print("\tGetting authors...")
    authors = np.unique(dataset[:, 0]).astype(int)

    print("\tGrouping examples...")
    grouped_examples = {}
    for author_id in authors:
        grouped_examples[author_id] = np.where((dataset[:, 0] == author_id))[0]

    texts = blogs["text"].to_numpy()
    os.makedirs("datasets/authors", exist_ok=True)
    with Pool(4) as p:
        p.map(make_csv_for_author, [(author, texts[indices].tolist(), tokenizer, bert) for author, indices in grouped_examples.items()])


@torch.no_grad()
def make_csv_for_author(x):
    author, texts_by_author, tokenizer, bert = x
    print(f"\tTokenizing.. for author {author}")
    data = tokenizer(texts_by_author, return_tensors="np", padding='max_length', truncation=True)
    input_ids = torch.from_numpy(data["input_ids"])
    attention_mask = torch.from_numpy(data["attention_mask"])

    print(f"\tGetting embeddings for author {author}...")
    x = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Now x is a tensor of shape (batch_size, sequence_length, hidden_size)
    # I want x to be (batch_size, hidden_size=768, sequence_length=512)
    x = x.transpose(1, 2)
    # cep

    print(f"\tGetting spectrogram for author {author}...")
    pos_spectral = Spectrogram(n_fft=N_FFT)
    x = pos_spectral(x)
    # Shape: (batch_size, hidden_size=768, c=fft_out=257, n_frames = 3)
    x = x.mean(3)
    # Shape: (batch_size, hidden_size=768, c=257 * 3)
    x = x.flatten(1)
    filename = f"datasets/authors/{author}.csv"
    print(f"\tSaving to {filename}...")
    pd.DataFrame(x.numpy()).to_csv(filename, index=False)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained(Path("models/bert"))
    main()