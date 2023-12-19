import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from constants import *
import swifter

class Matcher(Dataset):
    def __init__(self, blogs: DataFrame, train: bool, tokenizer):
        super().__init__()

        def random_truncate(text):
            split = text.split()
            length = len(split)
            if length >= 512:
                start = np.random.randint(length - 511)
                text = ' '.join(split[start:start+511])
            return text


        # get the train or test splits
        s = time.time()
        print("\tGetting split...")
        blogs = blogs[(blogs["split"] == "train") == train].drop("split", axis=1).reset_index(drop=True)
        s = report_time(s)
        print("\tTo numpy...")
        dataset = blogs.to_numpy()
        s = report_time(s)
        print("\tGetting authors...")
        self.authors = np.unique(dataset[:, 0]).astype(int)
        s = report_time(s)

        print("\tTruncating...")
        texts = blogs["text"].swifter.apply(random_truncate).tolist()
        s = report_time(s)

        print("\tTokenizing...")
        self.data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True)
        s = report_time(s)

        print("\tGrouping examples...")
        self.grouped_examples = {}
        for author_id in self.authors:
            self.grouped_examples[author_id] = np.where((dataset[:, 0] == author_id))[0]

        report_time(s)

    def __len__(self):
        return 64

    def __getitem__(self, index):
        """
            For every example, we will select two blogs. There are two cases,
            positive and negative examples. For positive examples, we will have two
            blogs from the same author. For negative examples, we will have two blogs
            from different authors.

            Given an index, if the index is even, we will pick the second blog from the same class,
            but it won't be the same blog we chose for the first author. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same blogs. However,
            if the network were given two different blogs from the same class, the network will need to learn
            the similarity between two different blogs from the same author. If the index is odd, we will
            pick the second blog from a different author than the first blog.
        """

        # pick a random author for the first blog
        selected_author = random.choice(self.authors)

        # pick a random index for the first image in the grouped indices
        # based of the label of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_author].shape[0] - 1)

        # pick the index to get the first blog
        index_1 = self.grouped_examples[selected_author][random_index_1]

        # get the first blog
        # blog contained in /datasets/authors/{author_id}.csv at index {index_1}
        blogs_by_author = pd.read_csv(f"datasets/authors/{selected_author}.csv")
        text_1 = blogs_by_author.iloc[index_1]["text"]

        # same class
        if index % 2 == 0:
            # pick a random index for the second blog
            random_index_2 = random.randint(0, self.grouped_examples[selected_author].shape[0] - 1)

            # ensure that the index of the second blog isn't the same as the first blog
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_author].shape[0] - 1)
                print("same index")

            # pick the index to get the second blog
            index_2 = self.grouped_examples[selected_author][random_index_2]

            # get the second blog
            text_2 = self.data[index_2]

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_author = random.choice(self.authors)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_author == selected_author:
                other_selected_author = random.choice(self.authors)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_author].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_author][random_index_2]

            # get the second image
            text_2 = self.data[index_2]

            # set the label for this example to be negative (-1)
            target = torch.tensor(-1, dtype=torch.float)

        text_1_ids = torch.tensor(text_1.ids)
        text_1_mask = torch.tensor(text_1.attention_mask)

        text_2_ids = torch.tensor(text_2.ids)
        text_2_mask = torch.tensor(text_2.attention_mask)

        return (text_1_ids, text_1_mask), (text_2_ids, text_2_mask), target

