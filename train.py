from prepare_data import prepare_data
from models import Baseline, UniLSTM, BiLSTM, BiLSTMMax, Model
import torch
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import spacy
import sys
import argparse


def seed_everything(seed: int):
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


class SentenceClassification:
    def __init__(self, sent_encoder_model: str, lr: float, epochs: int, seed: int):
        # configs
        self.data_path = Path("data/")
        self.path_to_senteval = Path("../SentEval/")  # to import the package
        self.path_to_sent_eval_data = "../SentEval/data"
        self.path_to_glove = self.path_to_senteval / "GloVe/glove.840B.300d.txt"
        self.checkpoint_path = Path("checkpoints/")
        self.log_path = Path("logs/")
        self.seed = seed
        self.lr = lr

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.batch_size = 256
        self.val_frequency = 100000
        self.verbose = True

        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sent_encoder_model = sent_encoder_model  # TODO adapt to other models
        self.criterion = torch.nn.CrossEntropyLoss()

        # reproducibility
        seed_everything(self.seed)

        # prepare data dl
        # dl contains keys: premise: (text, len), hypothesis (text, len), label
        self.train_dl, self.val_dl, self.test_dl, self.vocab = prepare_data(data_path=self.data_path, batch_size=self.batch_size)

        # set the encoding method and parameters regarding the encoding
        self.embedding_size = self.set_sent_encoder()

        # training hyperparameters
        self.model = Model(self.sent_encoder, self.embedding_size, hidden_dim=512, output_dim=3)  # check specifics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.highest_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0

        # tensorboard
        config = {"encoder": self.sent_encoder_model, "batch_size": self.batch_size, "lr": self.lr, "device": self.device, "max_epochs": self.epochs, "seed": self.seed}
        self.writer = SummaryWriter(self.log_path / f"{config}")

    def set_sent_encoder(self):
        if self.sent_encoder_model == "baseline":
            self.sent_encoder = Baseline(self.vocab.vectors)
            self.embedding_size = self.vocab.vectors.shape[1]  # sentence embedding size

            return self.embedding_size

        elif self.sent_encoder_model == "unilstm":  # unidirectional LSTM
            self.embedding_size = 2048  # TODO check specifics; eventually rename to hidden_size
            self.sent_encoder = UniLSTM(embeddings=self.vocab.vectors, hidden_size=self.embedding_size, batch_size=self.batch_size, num_layers=1, device=self.device)  # TODO check specifics

            return self.embedding_size

        elif self.sent_encoder_model == "bilstm":
            self.embedding_size = 2 * 2048  # sentence embedding obtained from both directions
            self.sent_encoder = BiLSTM(embeddings=self.vocab.vectors, hidden_size=self.embedding_size, batch_size=self.batch_size, num_layers=1, device=self.device)

            return self.embedding_size

        elif self.sent_encoder_model == "bilstmmax":
            self.embedding_size = 2 * 2048
            self.sent_encoder = BiLSTMMax(embeddings=self.vocab.vectors, hidden_size=self.embedding_size, batch_size=self.batch_size, num_layers=1, device=self.device)

            return self.embedding_size

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

    def _infer(self, text: list, tokenizer: spacy):
        tokenized = [w.text.lower() for w in tokenizer(text)]

        indexed, length = [self.vocab.stoi[t] for t in tokenized], [len(tokenized)]

        tensorized = torch.tensor(indexed).to(self.device)
        tensorized = tensorized.unsqueeze(1).T  # batchsize, seq_len
        length_tensor = torch.tensor(length, dtype=torch.long).to(self.device)

        return tensorized, length_tensor

    def infer(self, premise: str, hypothesis: str):
        self.model.eval()

        # tokenize, lowercase
        tokenizer = spacy.load("en_core_web_sm")

        premise, len_p = self._infer(premise, tokenizer)
        hypothesis, len_h = self._infer(hypothesis, tokenizer)

        # premise = torch.tensor(premise).to(self.device)
        # hypothesis = torch.tensor(hypothesis).to(self.device)

        output = self.model(premise, len_p, hypothesis, len_h)

        scores = torch.argmax(output, dim=1)  # shape: (batch_size,)

        return {"output": output, "scores": scores}

    def test_model(self):
        # load best model
        checkpoint = torch.load(self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # recover training hyperparameters # TODO really needed?
        self.best_val_acc = checkpoint["accuracy"]
        self.highest_epoch = checkpoint["epoch"]

        self.model.to(self.device)
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
        self.writer.add_scalar(f"loss/{mode}", loss, self.global_step)
        self.writer.add_scalar(f"accuracy/{mode}", accuracy, self.global_step)

        if accuracy > self.best_val_acc and mode == "val":
            torch.save(
                {
                    "epoch": self.highest_epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "accuracy": accuracy,
                },
                self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt",
            )
            self.best_val_acc = accuracy  # update best val acc

            print(f"New best model saved with accuracy: {accuracy:.4f} and loss: {loss:.4f} (at epoch: {self.highest_epoch}")

        return {"loss": loss, "accuracy": accuracy}

    def fit(self, resume_training=True):
        if resume_training:
            # load best model

            checkpoint_path_best_model = self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt"

            if Path.exists(checkpoint_path_best_model):
                checkpoint = torch.load(checkpoint_path_best_model)

                print(f"Resuming training from checkpoint {checkpoint_path_best_model}")

                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # recover training hyperparameters
                self.best_val_acc = checkpoint["accuracy"]
                self.highest_epoch = checkpoint["epoch"]
                self.global_step = checkpoint["global_step"]

                print(f"Found checkpoint with accuracy: {self.best_val_acc:.4f} at epoch: {self.highest_epoch}")

        else:  # start training from scratch
            print("Training from scratch")

            # vars for saving the model, and logging

        self.model.to(self.device)

        for epoch in range(self.highest_epoch, self.epochs):
            print(f"Epoch {epoch}/{self.epochs}")
            for b, batch in enumerate(self.val_dl):  # TOdo change to train_dl
                batch_results = self.train_batch(batch)

                # logging
                self.writer.add_scalar("loss/train", batch_results["loss"], self.global_step)
                self.writer.add_scalar("accuracy/train", batch_results["accuracy"], self.global_step)

            # validate model after each epoch on dev set
            self.evaluate_model(mode="val")
            self.highest_epoch += 1

    def prepare(self, params, samples):
        params.vocab = self.vocab
        params.encoder = self.sent_encoder  # use the same encoder as the model to encode the sentences
        return

    def batcher(self, params, batch):
        # assume batch already preprocessed
        sent_reps = []

        for sent in batch:
            if sent == []:
                sent = ["."]

            indexed = [params.vocab.stoi[word] for word in sent]
            tensorized = torch.tensor([indexed])
            len_tokens = len(tensorized)

            with torch.no_grad():
                sent_encoded = params.encoder(tensorized, len_tokens)

            sent_reps.append(sent_encoded.detach().numpy())

        sent_reps = np.vstack(sent_reps)

        return sent_reps

    def run_senteval(self):
        sys.path.insert(0, self.path_to_senteval)
        import senteval
        from warnings import simplefilter
        from sklearn.exceptions import ConvergenceWarning

        simplefilter("ignore", category=ConvergenceWarning)

        params_senteval = {"task_path": self.path_to_sent_eval_data, "usepytorch": False, "kfold": 5}
        params_senteval["classifier"] = {"nhid": 0, "optim": "adam", "batch_size": 128, "tenacity": 2, "epoch_size": 3}
        transfer_tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "SICKEntailment", "MRPC"]

        se = senteval.engine.SE(params_senteval, self.batcher, self.prepare)
        results = se.eval(transfer_tasks)

        return results


# training
def main(sent_encoder_model, lr, max_epochs, seed, mode, resume_training):
    # Done add argparse
    # TODO decrease lr
    # TODO check if same params as paper are used

    trainer = SentenceClassification(sent_encoder_model=sent_encoder_model, lr=lr, epochs=max_epochs, seed=seed)

    if mode == "train" or mode == "all":
        trainer.fit(resume_training=resume_training)

    if mode == "test" or mode == "all":
        trainer.test_model()

    if mode == "infer":
        # example for inference
        output = trainer.infer("The cat is on the mat", "The cat is on the mat")

        print(f'Predicted Class: {output["prediction"]}')
        print(f'Scores: {output["scores"]}')

    if mode == "senteval" or mode == "all":
        # test usign senteval
        results = trainer.run_senteval()
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sent_encoder_model", type=str, required=True, help="Sentence encoder model to use: baseline, unilstm, bilstm, orbilstmmax")
    parser.add_argument("--mode", type=str, default="all", help="Mode to run: train, test, infer, senteval, or all; default is all")
    parser.add_argument("--resume_training", type=bool, default=True, help="Whether to resume training from checkpoint or not, default True")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate, default 0.001")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs to train for, default 20")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42")

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
