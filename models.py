import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):
    def __init__(self, sentence_encoder, encoding_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder_block = sentence_encoder

        self.classifier = nn.Sequential(nn.Linear(2 * encoding_dim, hidden_dim), nn.Linear(hidden_dim, output_dim))  # TODO check specifics

    def forward(self, premise, len_premise, hypothesis, len_hypothesis):
        u = self.encoder_block(premise)
        v = self.encoder_block(hypothesis)

        # combinations of u and v
        abs_difference = torch.abs(u - v)
        product = u * v  # element-wise product!

        # concatenate
        concatenated = torch.cat((u, v, abs_difference, product), dim=1)

        output = self.classifier(concatenated)

        return output


class Baseline(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def forward(self, text):
        embedded = self.embedding(text[0])
        mean_embedded = torch.mean(embedded, dim=1)

        return mean_embedded


class LSTM(nn.Module):
    ...


class BiLSTM(nn.Module):
    ...


class BiLSTMMax(nn.Module):
    ...
