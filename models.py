import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):
    def __init__(self, sentence_encoder, encoding_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder_block = sentence_encoder

        # 4096 * 4 = 16384
        # print(f"Classifier expected in dimension: {4 * encoding_dim}, hidden_dim: {hidden_dim}, output_dim: {output_dim}")
        self.classifier = nn.Sequential(nn.Linear(4 * encoding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=1))  # TODO check specifics

    def forward(self, premise, len_premise, hypothesis, len_hypothesis):
        u = self.encoder_block(premise, len_premise)  # shape: (batch_size, encoding_dim)
        v = self.encoder_block(hypothesis, len_hypothesis)

        # combinations of u and v
        abs_difference = torch.abs(u - v)
        product = u * v  # element-wise product!

        # concatenate
        concatenated = torch.cat((u, v, abs_difference, product), dim=1)  # shape: (batch_size, 4 * encoding_dim)

        # print(f"Shape of concatenated: {concatenated.shape}")  # BiLSTM: [256, 32768]; UniLSTM: [256, 8192])

        output = self.classifier(concatenated)

        return output


class Baseline(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def forward(self, text, text_length):
        embedded = self.embeddings(text)
        mean_embedded = torch.mean(embedded, dim=1)  # mean over the embedding dimension: shape: (batch_size, embedding_dim=300)

        return mean_embedded


class UniLSTM(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers):
        super().__init__()
        input_size = embeddings.shape[1]  # embedding dimensionality

        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.layers = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, text, text_length):
        embedded = self.embeddings(text)

        # padding and packing
        embedded_padded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)  # batch, seq, feature
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded_padded, lengths=text_length, batch_first=True, enforce_sorted=False)  # optimize for speed

        # output: Batch_size,L,D*H out (packed sequence)
        # h_n: 1,Batch_size,H_out​ (1 = Direction*num_layers)
        # c_n: 1,Batch_size,H of cell (1 = D*num_layers)
        output, (h_n, c_n) = self.layers(embedded_packed)

        # print(f"h_n shape: {h_n.shape}")  # 1, 256, 2048])
        # print(f"c_n shape: {c_n.shape}")

        sent_repr = h_n.squeeze()  # remove the first dimension (num_layers)
        # print(f"sent_repr shape: {sent_repr.shape}")  # [256, 2048]
        return sent_repr


class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers):
        super().__init__()

        hidden_size = int(hidden_size / 2)  # hidden size is doubled because of bidirectional LSTM
        input_size = embeddings.shape[1]  # embedding dimensionality (300)

        # print(f"Input size: {input_size}")
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.layers = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)  # bidirectional LSTM

    def forward(self, text, text_length):
        embedded = self.embeddings(text)

        # padding and packing
        embedded_padded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)  # batch, seq, feature
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded_padded, lengths=text_length, batch_first=True, enforce_sorted=False)  # optimize for speed

        # output: Batch_size,L,D*H out (packed sequence)
        # h_n: 1,Batch_size,H_out​ (2 = D*num_layers)
        # c_n: 1,Batch_size,H of cell (2 = D*num_layers)
        output, (h_n, c_n) = self.layers(embedded_packed)

        # print(f"Output ty: {type(output)}")
        # print(f"h_n shape: {h_n.shape}")  # [2, 256, 4096]
        # print(f"c_n shape: {c_n.shape}")

        # combine both directions
        sent_repr = torch.cat((h_n[0], h_n[1]), dim=1)  # shape: (batch_size, 2 * hidden_size)
        # print(f"sent_repr shape: {sent_repr.shape}")  # [256, 8192])
        sent_repr = sent_repr.squeeze()  # remove the first dimension (num_layers)
        # print(f"sent_repr shape squeezed: {sent_repr.shape}")
        return sent_repr


class BiLSTMMax(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers):
        super().__init__()

        hidden_size = int(hidden_size / 2)  # hidden size is doubled because of bidirectional LSTM
        input_size = embeddings.shape[1]  # embedding dimensionality (300)

        # print(f"Input size: {input_size}")
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.layers = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)  # bidirectional LSTM

    def forward(self, text, text_length):
        embedded = self.embeddings(text)

        # padding and packing
        embedded_padded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)  # batch, seq, feature
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded_padded, lengths=text_length, batch_first=True, enforce_sorted=False)  # optimize for speed

        # output: Batch_size,L,D*H out (packed sequence)
        # h_n: 1,Batch_size,H_out​ (2 = D*num_layers)
        # c_n: 1,Batch_size,H of cell (2 = D*num_layers)
        output, (h_n, c_n) = self.layers(embedded_packed)

        # revert the packing
        output_padded_only = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]  # discards the lengths

        # find the max in all hidden states
        sent_repr_max_pooled = torch.max(output_padded_only, dim=1)[0]  # to get values only, discard indices

        return sent_repr_max_pooled
