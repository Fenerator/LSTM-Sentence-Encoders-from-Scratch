from datasets import load_dataset
import torch
import multiprocessing as mp
import nltk
import numpy as np

nltk.download("punkt")


def get_word2embedding(glove_file="data/GloVe/glove.840B.300d.txt"):
    word2embedding = {}
    with open(glove_file, "r") as f:
        for line in f:
            l = line.split()
            word = l[0]
            embedding = np.asarray(l[1:], "float32")
            word2embedding[word] = embedding

    return word2embedding


def to_embedding(tokens: list):
    embeddings = torch.empty((len(tokens), 300), dtype=torch.float32)
    for t, token in enumerate(tokens):
        if token in word2embedding:
            embeddings[t] = word2embedding[token]
        else:
            raise ValueError(f"Token {token} not in word2embedding")

    return embeddings


def preprocess(text: str):
    tokenized = nltk.tokenize.word_tokenize(text.lower())

    sent_tensor = to_embedding(tokenized)

    return {"premise": sent_tensor}


def prepare_snli_data(split="train"):
    # get embeddings
    global word2embedding
    word2embedding = get_word2embedding()

    # get dataset
    ds = load_dataset("snli", split=split)

    # preprocessing: lowercase, tokenize
    ds2 = ds.map(lambda x: preprocess(x["premise"]), batched=False)
    ds2 = ds2.map(lambda x: preprocess(x["hypothesis"]), batched=False)

    ds2.set_format(type="torch", columns=["premise", "hypothesis", "label"])

    # preprocessing

    dataloader = torch.utils.data.DataLoader(ds2, batch_size=1)

    print(next(iter(dataloader)))
