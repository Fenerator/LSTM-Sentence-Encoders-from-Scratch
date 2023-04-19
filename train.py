from prepare_data import prepare_data
from models import Baseline, UniLSTM, BiLSTM, BiLSTMMax, Model
import torch
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int):
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


class Train:
    def __init__(self, sent_encoder_model: str):
        # configs
        self.data_path = Path("data/")
        self.checkpoint_path = Path("checkpoints/")
        self.log_path = Path("logs/")
        self.seed = 42
        self.lr = 0.001

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.batch_sizes = (16, 256, 256)
        self.val_frequency = 100000
        self.verbose = True

        self.epochs = 2

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sent_encoder_model = sent_encoder_model  # TODO adapt to other models

        # reproducibility
        seed_everything(self.seed)

        # prepare data dl
        # dl contains keys: premise: (text, len), hypothesis (text, len), label
        self.train_dl, self.val_dl, self.test_dl, self.vocab = prepare_data(data_path=self.data_path, batch_sizes=self.batch_sizes)

        # set the encoding method and parameters regarding the encoding
        self.set_sent_encoder()

        # training hyperparameters
        self.model = Model(self.sent_encoder, self.embedding_size, hidden_dim=512, output_dim=3)  # check specifics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

        # tensorboard
        config = {"encoder": self.sent_encoder_model, "val_freq": self.val_frequency, "batch_size": self.batch_sizes, "lr": self.lr, "device": self.device, "epochs": self.epochs, "seed": self.seed}
        self.writer = SummaryWriter(self.log_path / f"{config}")

    def set_sent_encoder(self):
        if self.sent_encoder_model == "baseline":
            self.sent_encoder = Baseline(self.vocab.vectors)
            self.embedding_size = self.vocab.vectors.shape[1]  # sentence embedding size

        elif self.sent_encoder_model == "unilstm":  # unidirectional LSTM
            self.embedding_size = 2048  # TODO check specifics; eventually rename to hidden_size
            self.sent_encoder = UniLSTM(embeddings=self.vocab.vectors, hidden_size=self.embedding_size, batch_size=self.batch_sizes, num_layers=1, device=self.device)  # TODO check specifics

        elif self.sent_encoder_model == "bilstm":
            ...
            self.embedding_size = ...

        elif self.sent_encoder_model == "bilstmmax":
            ...
            self.embedding_size = ...

        else:
            raise ValueError(f"Model {self.sent_encoder_model} not implemented.")

    def train_batch(self, batch):
        self.global_step += 1  # for tensorboard
        batch_results = {}

        self.model.train()
        self.optimizer.zero_grad()

        # get the data elements
        premise, len_premise = batch.premise[0].to(self.device), batch.premise[1].to(self.device)
        hypothesis, len_hypothesis = batch.hypothesis[0].to(self.device), batch.hypothesis[1].to(self.device)
        labels = batch.label.to(self.device)

        # forward pass
        output = self.model(premise, len_premise, hypothesis, len_hypothesis)

        predictions = torch.argmax(output, dim=1)  # shape: (batch_size,)

        # calculate loss
        loss = self.criterion(output, labels)

        # backprop
        loss.backward()
        self.optimizer.step()

        # metrics
        batch_results["loss"] = loss.item()
        batch_results["accuracy"] = (predictions == labels).float().mean()

        return batch_results

    def val_batch(self, batch):
        batch_results = {}

        # get the data elements
        premise, len_premise = batch.premise[0].to(self.device), batch.premise[1].to(self.device)
        hypothesis, len_hypothesis = batch.hypothesis[0].to(self.device), batch.hypothesis[1].to(self.device)
        labels = batch.label.to(self.device)

        # forward pass
        output = self.model(premise, len_premise, hypothesis, len_hypothesis)
        predictions = torch.argmax(output, dim=1)  # shape: (batch_size,)

        # calculate loss
        loss = self.criterion(output, labels)

        batch_results["loss"] = loss.item()
        batch_results["accuracy"] = (predictions == labels).float().mean()

        return batch_results

    def test_model(self):
        # load best model
        checkpoint = torch.load(self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # recover training hyperparameters # TODO really needed?
        self.best_val_acc = checkpoint["accuracy"]
        self.highest_epoch = checkpoint["epoch"]

        eval_results = self.evaluate_model(mode="test")

        print(f"Test loss: {eval_results['loss']:.4f}, Test accuracy: {eval_results['accuracy']:.4f}")

    def evaluate_model(self, mode="val"):
        batch_losses, batch_accuracies = [], []

        dl = self.val_dl if mode == "val" else self.test_dl

        self.model.eval()
        with torch.no_grad():
            for b, batch in enumerate(dl):
                batch_results = self.val_batch(batch)
                batch_accuracies.append(batch_results["accuracy"])
                batch_losses.append(batch_results["loss"])

        # calculate metrics per epoch
        loss = sum(batch_losses) / len(batch_losses)
        accuracy = sum(batch_accuracies) / len(batch_accuracies)

        # logging
        self.writer.add_scalar(f"epoch_loss/{mode}", loss, self.global_step)
        self.writer.add_scalar(f"epoch_accuracy/{mode}", accuracy, self.global_step)

        if accuracy > self.best_val_acc and mode == "val":
            torch.save(
                {
                    "epoch": self.highest_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "accuracy": accuracy,
                },
                self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt",
            )
            self.best_val_acc = accuracy  # update best val acc

            print(f"New best model saved with accuracy: {accuracy:.4f} and loss: {loss:.4f}! (at epoch: {self.highest_epoch}")

        return {"loss": loss, "accuracy": accuracy}

    def train_model(self):
        # vars for saving the model, and logging
        self.highest_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0
        self.current_epoch = 0  # TODO really needed?

        for epoch in range(0, self.epochs):
            for b, batch in enumerate(self.train_dl):
                batch_results = self.train_batch(batch)

                # logging
                self.writer.add_scalar("loss/train", batch_results["loss"], self.global_step)
                self.writer.add_scalar("accuracy/train", batch_results["accuracy"], self.global_step)

                # validate model on dev set
                if b % self.val_frequency == 0:
                    self.evaluate_model(mode="val")

            self.current_epoch += 1  # TODO really needed?
            self.highest_epoch += 1


# training
def main():
    # TODO add argparse
    trainer = Train(sent_encoder_model="unilstm")
    trainer.train_model()
    trainer.test_model()


if __name__ == "__main__":
    main()
