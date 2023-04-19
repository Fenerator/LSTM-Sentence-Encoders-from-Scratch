import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):
    def __init__(self, sentence_encoder, encoding_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder_block = sentence_encoder

        self.classifier = nn.Sequential(nn.Linear(4 * encoding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))  # TODO check specifics

    def forward(self, premise, len_premise, hypothesis, len_hypothesis):
        u = self.encoder_block(premise)  # shape: (batch_size, encoding_dim)
        v = self.encoder_block(hypothesis)

        # combinations of u and v
        abs_difference = torch.abs(u - v)
        product = u * v  # element-wise product!

        # concatenate
        concatenated = torch.cat((u, v, abs_difference, product), dim=1)  # shape: (batch_size, 4 * encoding_dim)

        # print(f"Shape of concatenated: {concatenated.shape}")

        output = self.classifier(concatenated)

        # print(f"Shape of output: {output.shape}")
        return output


class Baseline(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def forward(self, text):
        embedded = self.embeddings(text)
        mean_embedded = torch.mean(embedded, dim=1)  # mean over the embedding dimension: shape: (batch_size, embedding_dim=300)

        return mean_embedded


class UniLSTM(nn.Module):
    def __init__(self, embeddings, hidden_size, batch_size, num_layers, device):
        super().__init__()
        input_size = embeddings.shape[1]

        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        print(f"Input size: {input_size}")

        self.layers = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

        # tensors for the initial hidden and cell states
        self.h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)  # todo rename!!!!!
        self.c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)  # todo rename!!!!!

    def forward(self, text, text_length):
        embedded = self.embeddings(text)

        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths=text_length, batch_first=True, enforce_sorted=False)
        x, (hn, cn) = self.layers(embedded_packed, (self.h0, self.c0))  # TODO rename x

        last_hn = hn[-1]  # return the last hidden state of the last layer # TODO same as squeeze()?

        return last_hn


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text):
        ...


class BiLSTMMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text):
        ...
