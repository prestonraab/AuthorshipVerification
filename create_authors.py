import numpy as np
import pandas as pd
from pathlib import Path
import swifter
import os
CHUNK_SIZE=1000
from transformers import AutoTokenizer
from constants import *


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

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("\tTruncating...")
    texts = blogs["text"].swifter.apply(random_truncate).tolist()

    print("\tTokenizing...")
    data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True)

    dataset = blogs.to_numpy()

    # find the blogs corresponding to each author
    # Then save those tokens to a file
    for author, blogs in blogs.groupby("id"):
        blogs = blogs[["text"]]
        author = str(author)
        # make new csv file
        os.makedirs("datasets/authors", exist_ok=True)
        filename = f"datasets/authors/{author}.csv"
        blogs.to_csv(filename, index=False)


if __name__ == '__main__':
    main()