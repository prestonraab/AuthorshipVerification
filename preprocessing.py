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
import tqdm

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
    i = 0
    for author_id in tqdm.tqdm(sorted(authors)):
        grouped_examples[str(i := i+1)] = np.where((dataset[:, 0] == author_id))[0]

    texts = blogs["text"].to_numpy()
    os.makedirs("datasets/authors", exist_ok=True)

    with Pool(3) as p:
        p.map(make_file_for_author, [(author, texts[indices].tolist(), tokenizer, bert) for author, indices in grouped_examples.items()])


@torch.no_grad()
def make_file_for_author(x):
    author, texts_by_author, tokenizer, bert = x
    if len(texts_by_author) > 20:
        make_file_for_author((author + "_even_" , texts_by_author[::2], tokenizer, bert))
        make_file_for_author((author + "_odd_" , texts_by_author[1::2], tokenizer, bert))
        return

    print(f"\tTokenizing.. for author {author} / 8965 {int(author.split('_')[0])/ 8965 * 100:.2f}%")
    params = {'return_tensors': 'pt', 'padding': 'max_length', 'truncation': True}
    torch.save(Spectrogram(n_fft=N_FFT)(bert(**tokenizer(texts_by_author, **params)).last_hidden_state.transpose(1, 2)).mean(3).flatten(1), f"datasets/authors/{author}.pt")


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained(Path("models/bert"))
    main()