from datasets import load_dataset
import torch
import multiprocessing as mp
import nltk
import numpy as np
import pandas as pd
import csv
import os, pickle

nltk.download("punkt")


def load_glove_model(glove_file="data/GloVe/glove.840B.300d.txt"):
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
                emb = np.array(list(map(float, vec[1:])), dtype=np.float32)
                embeddings[vocab_word] = emb

        with open(glove_file + ".cache", "wb") as cache_file:
            pickle.dump(embeddings, cache_file)

        print(f"Loaded {len(embeddings)} GloVe embeddings.")

    return embeddings
    # word2embedding = {}
    # print(f"Loading GloVe embeddings from {glove_file}...")
    # with open(glove_file, "r") as f:
    #     for line in f:
    #         l = line.split()
    #         print(f"word: {l[0]}")
    #         print(f"embedding: {l[1:5]}")
    #         word = l[0]
    #         embedding = np.array(l[1:], dtype=np.float64)
    #         word2embedding[word] = embedding
    # print(f"Loaded {len(word2embedding)} embeddings.")

    return word2embedding


def to_embedding(tokens: list):
    embeddings = torch.empty((len(tokens), 300), dtype=torch.float32)
    for token in tokens:
        if token in word2embedding.index:
            embeddings[0] = word2embedding.loc[token].as_matrix()
        else:
            raise ValueError(f"Token {token} not in word2embedding")

    return embeddings


def preprocess(text: str, column: str = "premise"):
    print(f"text: {text}")
    tokenized = nltk.tokenize.word_tokenize(text.lower())
    print(f"tokenized: {tokenized}")
    sent_tensor = to_embedding(tokenized)
    print(f"sent_tensor: {sent_tensor}")

    return {f"'{column}': {sent_tensor}"}


def prepare_snli_data(split="train"):
    # get embeddings
    global word2embedding
    word2embedding = load_glove_model()

    # get dataset
    ds = load_dataset("snli", split=split)

    # preprocessing: lowercase, tokenize
    ds2 = ds.map(lambda x: preprocess(x["premise"]), batched=False)
    # ds2 = ds2.map(lambda x: preprocess(x["hypothesis"]), batched=False)

    ds2.set_format(type="torch", columns=["premise", "hypothesis", "label"])

    # preprocessing

    dataloader = torch.utils.data.DataLoader(ds2, batch_size=1)

    print(next(iter(dataloader)))
