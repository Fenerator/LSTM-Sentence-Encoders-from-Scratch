from prepare_data import prepare_data
from models import Baseline, LSTM, BiLSTM, BiLSTMMax, Model
import torch
from pathlib import Path


class Train:
    def __init__(self, sent_encoder_model: str):
        # configs
        self.data_path = Path("data/")
        self.checkpoint_path = Path("checkpoints/")

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.batch_sizes = (16, 256, 256)
        self.val_frequency = 10000
        self.verbose = True

        self.epochs = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sent_encoder_model = sent_encoder_model  # TODO adapt to other models

        # prepare data dl
        # dl contains keys: premise: (text, len), hypothesis (text, len), label
        self.train_dl, self.val_dl, self.test_dl, self.vocab = prepare_data(data_path=self.data_path, batch_sizes=self.batch_sizes)

        # set the encoding method
        if self.sent_encoder_model == "baseline":
            self.sent_encoder = Baseline(self.vocab.vectors)
            self.encoding_dim = self.vocab.vectors.shape[1]

        # training hyperparameters
        self.model = Model(self.sent_encoder, self.encoding_dim, hidden_dim=512, output_dim=3)  # check specifics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

    def train_batch(self, batch):
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

        batch_results["loss"] = loss.item()

        # log acc of the batch # TODO
        print(f"Shape of output: {output.shape}, shape of labels: {labels.shape}, shape of predictions: {predictions.shape}")
        print(f"Output: {output}, labels: {labels}, predictions: {predictions}")

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

        # log acc of the batch # TODO
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

        eval_results = self.evaluate_model(self.test_dl, mode="test")

        print(f"Test loss: {eval_results['loss']:.4f}, Test accuracy: {eval_results['accuracy']:.4f}")

    def evaluate_model(self, mode="validation"):
        batch_losses, batch_accuracies = [], []

        dl = self.val_dl if mode == "validation" else self.test_dl

        self.model.eval()
        with torch.no_grad():
            for b, batch in enumerate(dl):
                batch_results = self.val_batch(batch)
                batch_accuracies.append(batch_results["accuracy"])
                batch_losses.append(batch_results["loss"])

        loss = sum(batch_losses) / len(batch_losses)
        accuracy = sum(batch_accuracies) / len(batch_accuracies)

        if accuracy > self.best_val_acc and mode == "validation":
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

        return {"loss": loss, "accuracy": accuracy}

        # TODO logging

    def train_model(self):
        self.highest_epoch = 0  # needed for saving the model
        self.best_val_acc = 0.0

        for epoch in range(0, self.epochs):
            for b, batch in enumerate(self.train_dl):
                batch_results = self.train_batch(batch)  # TODO change verbose

                # print loss
                # if b % 100 == 0:
                if self.verbose:
                    print(f"Epoch: {epoch+1}, Batch: {b+1}, Loss: {batch_results['loss']:.4f}, Accuracy: {batch_results['accuracy']:.4f}")

                # TODO logging
                ...

                # validate model on dev set
                if b % self.val_frequency == 0:
                    self.evaluate_model(mode="validation")

            self.highest_epoch += 1


# training
def main():
    # TODO add argparse
    trainer = Train(sent_encoder_model="baseline")
    trainer.train_model()
    trainer.test_model()


if __name__ == "__main__":
    main()
