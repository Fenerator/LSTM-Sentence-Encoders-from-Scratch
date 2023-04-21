from datasets import load_dataset
import torch
import multiprocessing as mp
import nltk
import numpy as np
import pandas as pd
import csv
import os, pickle
from collections import Counter, OrderedDict, defaultdict
import torchtext

nltk.download("punkt")


class OrderedCounter(Counter, OrderedDict):
    """From NLP1, practical 2:
    Counter that remembers the order elements are first seen
    """

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """From NLP1, practical 2
    used to assign ids to tokens
    """

    def __init__(self, embedding_for_special_tokens=np.zeros(300)):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []
        self.i2embedding = []

        # Only add these once
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_embedding(embedding_for_special_tokens)  # embedding for <unk>

        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)
        self.add_embedding(embedding_for_special_tokens)  # embedding for <pad>

        print(f"Special tokens embedding dim: {embedding_for_special_tokens.shape}")

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def add_embedding(self, embedding):
        self.i2embedding.append(embedding)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                in the vocabulary
        """

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)

        # create a matrix of all embeddings
        self.vectors = np.stack(self.i2embedding)
        self.vectors = self.vectors.astype(np.float32)


class Custom_Dataset:
    def __init__(self, data_path, batch_size=1) -> None:
        self.data_path = data_path
        self.batch_size = batch_size

        # build vocabulary
        """self.vocab = self.load_glove_embeddings(self.glove_file)  # load vocab from glove embeddings

        # preporcessing
        print(f"Creating dataloader...")
        self.test_ds = self.prepare_snli_data(split="test")
        self.val_ds = self.prepare_snli_data(split="validation")
        # self.train_ds = self.prepare_snli_data(split="train") # TODO

        # create dataloaders
        self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.train_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)  # todo change
        """

    def load_glove_embeddings(self, glove_file="data/GloVe/glove.840B.300d.txt"):
        if os.path.exists(glove_file + "_vocab.cache"):
            print("Found Vocab cache. Loading...")
            with open(glove_file + "_vocab.cache", "rb") as cache_file:
                vocab = pickle.load(cache_file)

        else:
            vocab = Vocabulary(embedding_for_special_tokens=np.zeros(300))
            print("No Vocab cache found. Loading Vocab from GloVe embeddings file...")
            # embeddings = dict() old

            with open(glove_file) as f:
                for line in f:
                    line = line.rstrip("\n")
                    vocab_word, *vec = line.rsplit(" ", maxsplit=300)
                    assert len(vec) == 300, f"Unexpected line: {line}"

                    # emb = np.array(list(map(float, vec)), dtype=np.float32)
                    vocab.add_token(vocab_word)
                    vocab.add_embedding(np.array(list(map(float, vec)), dtype=np.float32))
                    # embeddings[vocab_word] = emb OLD

            # build vocab
            vocab.build()

            with open(glove_file + "_vocab.cache", "wb") as cache_file:
                pickle.dump(vocab, cache_file)

        print(f"Loaded {len(vocab.w2i)} GloVe embeddings. \n Embedding matrix shape: {vocab.vectors.shape}")

        return vocab

    def prepare_snli_data(self, split: str = "train"):
        if os.path.exists(self.data_path + f"_snli_{split}.cache"):
            print(f"Found {split} preprocessed cache. Loading...")
            with open(self.data_path + f"_snli_{split}.cache", "rb") as cache_file:
                ds2 = pickle.load(cache_file)

        else:
            # get dataset
            ds = load_dataset("snli", split=split)

            # preprocessing: lowercase, tokenize
            print(f"Preprocessing {split} snli data... premise")
            ds2 = ds.map(lambda x: self.preprocess(x["premise"], "premise"), batched=False)
            print(f"Preprocessing {split} snli data... hypothesis")
            ds2 = ds2.map(lambda x: self.preprocess(x["hypothesis"], "hypothesis"), batched=False)

            ds2.set_format(type="torch", columns=["premise", "hypothesis", "label"])

            with open(self.data_path + f"_snli_{split}.cache", "wb") as cache_file:
                pickle.dump(ds2, cache_file)

        return ds2

    def preprocess(self, text: str, column: str = "premise"):
        tokenized = nltk.tokenize.word_tokenize(text)
        tokenized = [t.lower() for t in tokenized]

        # sent_tensor = self.to_embedding(tokenized)
        # print(f"sent_tensor shape: {sent_tensor.shape}")

        return {f"{column}": tokenized}

    def to_embedding(self, tokens: list):
        embeddings = torch.zeros((len(tokens), 300), dtype=torch.float32)

        for t, token in enumerate(tokens):
            if token in self.glove_embeddings.keys():
                token_emb = self.glove_embeddings[token]
                embeddings[t, :] = torch.from_numpy(token_emb).to(embeddings)  # token, embedding dim
            else:
                raise ValueError(f"Token {token} not in vocabulary of GloVe.")

        return embeddings
