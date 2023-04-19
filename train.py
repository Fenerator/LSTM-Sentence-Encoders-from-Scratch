from prepare_data import prepare_data
from models import Baseline, LSTM, BiLSTM, BiLSTMAttention, Model
import torch


# prepare data
class Train:
    def __init__(self, data_path: str, batch_sizes: tuple):
        # configs
        self.data_path = "data/"
        self.batch_sizes = (16, 256, 256)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sent_encoder_model = "baseline"  # TODO adapt to other models

        # prepare data
        self.train_dl, self.val_dl, self.test_dl, self.vocab = prepare_data(data_path=self.data_path, batch_sizes=self.batch_sizes)

        # set the encoding method
        if self.sent_encoder_model == "baseline":
            self.sent_encoder = Baseline(self.vocab.vectors)

        # training hyperparameters
        self.model = Model(self.sent_encoder, encoding_dim=300, hidden_dim=300, output_dim=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

    def train():
        ...


# training
# encode premise and hypothesis with GloVE embeddings

# encode premise and hypothesis with LSTM

# classify premise and hypothesis
# all in all 4 models


# evaluation
