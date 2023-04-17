from datasets import load_dataset
import torch
import multiprocessing as mp
import nltk
import numpy as np
import pandas as pd
import csv
import os, pickle

nltk.download("punkt")


class Custom_Dataset:
    def __init__(self, glove_path) -> None:

        self.glove_file = glove_path

        self.glove_embeddings = self.load_glove_model(self.glove_file)

        # preporcessing and creating dataloaders
        self.test_dl = self.prepare_snli_data(split="test")
        self.val_dl = self.prepare_snli_data(split="validation")
        self.train_dl = self.prepare_snli_data(split="train")

    def load_glove_model(self, glove_file="data/GloVe/glove.840B.300d.txt"):
        if os.path.exists(glove_file + ".cache"):
            print("Found cache. Loading...")
            with open(glove_file + ".cache", "rb") as cache_file:
                embeddings = pickle.load(cache_file)
        else:
            print("No cache found. Loading GloVe embeddings from file...")
            embeddings = dict()

            with open(glove_file) as f:
                for line in f:
                    line = line.rstrip("\n")
                    vocab_word, *vec = line.rsplit(" ", maxsplit=300)
                    assert len(vec) == 300, f"Unexpected line: {line}"
                    emb = np.array(list(map(float, vec)), dtype=np.float32)
                    embeddings[vocab_word] = emb

            with open(glove_file + ".cache", "wb") as cache_file:
                pickle.dump(embeddings, cache_file)

        print(f"Loaded {len(embeddings)} GloVe embeddings.")

        return embeddings

    def to_embedding(self, tokens: list):
        embeddings = torch.zeros((len(tokens), 300), dtype=torch.float32)

        for t, token in enumerate(tokens):
            if token in self.glove_embeddings.keys():
                token_emb = self.glove_embeddings[token]
                embeddings[t, :] = torch.from_numpy(token_emb).to(embeddings)  # token, embedding dim
            else:
                raise ValueError(f"Token {token} not in vocabulary of GloVe.")

        return embeddings

    def preprocess(self, text: str, column: str = "premise"):
        print(f"text: {text}")
        tokenized = nltk.tokenize.word_tokenize(text.lower())
        print(f"tokenized: {tokenized}")
        sent_tensor = self.to_embedding(tokenized)
        print(f"sent_tensor shape: {sent_tensor.shape}")
        return {f"{column}": sent_tensor}

    def prepare_snli_data(self, split="train"):
        print(f"Loading {split} data...")

        # get dataset
        ds = load_dataset("snli", split=split)

        # preprocessing: lowercase, tokenize
        print(f"Preprocessing {split} data...")
        ds2 = ds.map(lambda x: self.preprocess(x["premise"]), batched=False)
        # ds2 = ds2.map(lambda x: preprocess(x["hypothesis"]), batched=False)

        ds2.set_format(type="torch", columns=["premise", "hypothesis", "label"])

        # preprocessing

        dataloader = torch.utils.data.DataLoader(ds2, batch_size=1)

        print(next(iter(dataloader)))
