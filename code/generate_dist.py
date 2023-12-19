import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel
from constants import *
from verifier import Verifier


def similarity (text1, text2):
    pass

"""
Generates two distribution of cosine similarity values
One for blogs by the same author
One for blogs by different authors
"""
def generate_distributions():
    # load model from siamese_network.pt
    verifier = Verifier().to("cpu")
    verifier.load_state_dict(torch.load("siamese_network.pt"))


    blogs = pd.read_csv("datasets/blog8965.csv.gz").dropna()
    blogs = blogs[blogs["text"].apply(str.split).apply(len) > 38]
    similarities = []
    for author, blogs in blogs.groupby("id"):
        for i, blog in enumerate(blogs["text"]):
            for j, other_blog in enumerate(blogs["text"]):
                if i != j:
                    similarities.append(similarity(blog, other_blog))

if __name__ == '__main__':
    generate_distributions()