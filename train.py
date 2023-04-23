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
    def __init__(self, sent_encoder_model: str, epochs: int, seed: int, resume_training: bool, verbose: bool, optimizer_name: str):
        # paths
        self.path_to_senteval = Path("../SentEval/")  # to import the package
        self.path_to_sent_eval_data = "../SentEval/data"  # senteval not working with pathlib
        self.checkpoint_path = Path("checkpoints/")
        self.log_path = Path("logs/")

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        if not Path.exists(self.path_to_senteval):
            raise FileNotFoundError(f"SentEval not found at expected location: {self.path_to_senteval}")

        # settings
        self.resume_training = resume_training
        self.verbose = verbose  # additional print statements

        # training hyperparameters
        self.seed = seed
        self.shrink_factor = 5.0
        self.batch_size = 64
        self.epochs = epochs
        self.min_lr = 1e-5  # only for sgd
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sent_encoder_model = sent_encoder_model
        self.criterion = torch.nn.CrossEntropyLoss()
        classifier_hidden_dim = 512
        classifier_output_dim = 3

        # initial training state
        self.highest_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0

        # reproducibility
        seed_everything(self.seed)

        # prepare data dl
        # dl contains keys: premise: (text, len), hypothesis (text, len), label
        self.train_dl, self.val_dl, self.test_dl, self.vocab = prepare_data(data_path=self.path_to_sent_eval_data, batch_size=self.batch_size)

        # set the encoding method and parameters regarding the encoding
        self.sent_embedding_size = self.set_sent_encoder()
        self.model = Model(self.sent_encoder, self.sent_embedding_size, hidden_dim=classifier_hidden_dim, output_dim=classifier_output_dim, device=self.device)

        self.optimizer_name = optimizer_name

        if self.optimizer_name == "sgd":
            self.lr = 1e-1
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if self.optimizer_name == "adam":
            self.lr = 1e-3  # seems not to be working with 0.1
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # tensorboard
        config = {
            "encoder": self.sent_encoder_model,
            "batch_size": self.batch_size,
            "sent_emb_dim": self.sent_embedding_size,
            "optimizer": self.optimizer_name,
            "lr": self.lr,
            "device": self.device,
            "max_epochs": self.epochs,
            "seed": self.seed,
        }
        print(f"Config: {config} \n Resume_training = {self.resume_training} \n verbose = {self.verbose}")

        self.writer = SummaryWriter(self.log_path / f"{config}")

    def set_sent_encoder(self):
        print(f"Using {self.sent_encoder_model} as sentence encoder")

        if self.sent_encoder_model == "baseline":
            self.sent_encoder = Baseline(self.vocab.vectors, device=self.device)
            self.sent_embedding_size = self.vocab.vectors.shape[1]  # sentence embedding size

            return self.sent_embedding_size

        elif self.sent_encoder_model == "unilstm":  # unidirectional LSTM
            self.sent_embedding_size = 2048
            self.sent_encoder = UniLSTM(embeddings=self.vocab.vectors, hidden_size=self.sent_embedding_size, num_layers=1, device=self.device)

            return self.sent_embedding_size

        elif self.sent_encoder_model == "bilstm":
            self.sent_embedding_size = 4096  # twice as large, as sentence embedding obtained from both directions
            self.sent_encoder = BiLSTM(embeddings=self.vocab.vectors, hidden_size=self.sent_embedding_size, num_layers=1, device=self.device)

            return self.sent_embedding_size

        elif self.sent_encoder_model == "bilstmmax":
            self.sent_embedding_size = 4096  # twice as large, as sentence embedding obtained from both directions
            self.sent_encoder = BiLSTMMax(embeddings=self.vocab.vectors, hidden_size=self.sent_embedding_size, num_layers=1, device=self.device)

            return self.sent_embedding_size

        else:
            raise ValueError(f"Model {self.sent_encoder_model} not implemented.")

    def train_batch(self, batch):
        self.global_step += 1  # for tensorboard
        batch_results = {}

        self.model.train()
        self.optimizer.zero_grad()

        # get the data elements
        # TODO remove: was like this premise, len_premise = batch.premise[0].to(self.device), batch.premise[1].to(self.device)
        premise, len_premise = batch.premise[0], batch.premise[1]
        hypothesis, len_hypothesis = batch.hypothesis[0], batch.hypothesis[1]
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
        premise, len_premise = batch.premise[0], batch.premise[1]
        hypothesis, len_hypothesis = batch.hypothesis[0], batch.hypothesis[1]
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

        tensorized = torch.tensor(indexed)
        tensorized = tensorized.unsqueeze(1).T  # batchsize, seq_len
        length_tensor = torch.tensor(length, dtype=torch.long)

        return tensorized, length_tensor

    def infer(self, premise: str, hypothesis: str):
        self.model.eval()

        # tokenize, lowercase
        tokenizer = spacy.load("en_core_web_sm")

        premise, len_p = self._infer(premise, tokenizer)
        hypothesis, len_h = self._infer(hypothesis, tokenizer)

        output = self.model(premise, len_p, hypothesis, len_h)

        predicted_label = torch.argmax(output, dim=1)  # shape: (batch_size,)

        return {"output": output, "label": predicted_label}

    def test_model(self):
        # load best model
        checkpoint = torch.load(self.checkpoint_path / f"{self.sent_encoder_model}_best_model.pt")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # recover training hyperparameters
        self.best_val_acc = checkpoint["accuracy"]
        self.highest_epoch = checkpoint["epoch"]

        self.model.to(self.device)
        eval_results = self.evaluate_model(mode="test")

        print(f"Test loss: {eval_results['loss']:.4f}, Test accuracy: {eval_results['accuracy']:.4f}")
        print(f"Best validation accuracy was: {self.best_val_acc:.4f} at epoch {self.highest_epoch}")

    def evaluate_model(self, mode="val"):
        improved = False
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
            improved = True
            torch.save(
                {
                    "epoch": self.highest_epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "accuracy": accuracy,
                },
                self.checkpoint_path / f"{self.sent_encoder_model}_{self.optimizer_name}_best_model.pt",
            )
            self.best_val_acc = accuracy  # update best val acc

            print(f"New best model saved with accuracy: {accuracy:.4f} and loss: {loss:.4f} (at epoch: {self.highest_epoch}")

        if self.verbose:
            print(f"loss/{mode}: {loss:.4f}, accuracy/{mode}: {accuracy:.4f}")

        return {"loss": loss, "accuracy": accuracy, "improved": improved}

    def fit(self, resume_training=True):
        if resume_training:
            # load best model

            checkpoint_path_best_model = self.checkpoint_path / f"{self.sent_encoder_model}_{self.optimizer_name}_best_model.pt"

            if Path.exists(checkpoint_path_best_model):
                checkpoint = torch.load(checkpoint_path_best_model, map_location="cpu")

                print(f"Resuming training from checkpoint {checkpoint_path_best_model} using optimizer: {self.optimizer_name}")

                self.model.to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # recover training hyperparameters
                self.best_val_acc = checkpoint["accuracy"]
                self.highest_epoch = checkpoint["epoch"]
                self.global_step = checkpoint["global_step"]

                print(f"Found checkpoint with val accuracy: {self.best_val_acc:.4f} at epoch: {self.highest_epoch}")

        else:  # start training from scratch
            print("Training from scratch")

            # vars for saving the model, and logging

            self.model.to(self.device)

        # TRAINING LOOP
        for epoch in range(self.highest_epoch, self.epochs):
            print(f"Training: Epoch {epoch}/{self.epochs}")
            for b, batch in enumerate(self.train_dl):
                batch_results = self.train_batch(batch)

                # logging
                self.writer.add_scalar("loss/train", batch_results["loss"], self.global_step)
                self.writer.add_scalar("accuracy/train", batch_results["accuracy"], self.global_step)

            # validate model after each epoch on dev set
            eval_results = self.evaluate_model(mode="val")
            self.highest_epoch += 1

            # decrease learning rate if improvement in val accuracy
            if eval_results["improved"]:
                if self.verbose and self.optimizer_name == "sgd":
                    print(f'Old learning rate: {self.optimizer.param_groups[0]["lr"]}')

                if self.optimizer_name == "sgd":
                    self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] / self.shrink_factor  # decrease lr if improvement

                    if self.verbose:
                        print(f"Learning rate decreased to: {self.optimizer.param_groups[0]['lr']}")

                    if self.optimizer.param_groups[0]["lr"] < self.min_lr:
                        print("========== Learning rate too small, stopping training ==========")
                        break

    def prepare(self, params, samples):
        params.vocab = self.vocab
        params.encoder = self.sent_encoder  # use the same encoder as the model to encode the sentences
        return

    def batcher(self, params, batch):
        # assume batch already preprocessed
        sent_reps = []

        for sent in batch:
            if sent == []:
                print(f"Empty sentence in batcher")
                sent = ["."]  # TODO check if needed

            indexed = [params.vocab.stoi[word] for word in sent]
            tensorized = torch.tensor([indexed])
            len_tokens = len(tensorized)

            with torch.no_grad():
                sent_encoded = params.encoder(tensorized, len_tokens)

            sent_reps.append(sent_encoded.detach().numpy())

        sent_reps = np.vstack(sent_reps)

        return sent_reps

    def run_senteval(self):
        self.device = torch.device("cpu")

        print(f"====== Running senteval on device {self.device}======")

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
def main(sent_encoder_model, max_epochs, seed, mode, resume_training, verbose, optimizer):
    trainer = SentenceClassification(sent_encoder_model=sent_encoder_model, epochs=max_epochs, seed=seed, resume_training=resume_training, verbose=verbose, optimizer_name=optimizer)

    if mode == "train" or mode == "all":
        trainer.fit(resume_training=resume_training)

    if mode == "test" or mode == "all":
        trainer.test_model()

    if mode == "infer":
        # example for inference
        output = trainer.infer("The cat is on the mat", "The cat is on the mat")

        print(f'Predicted Class: {output["label"]}')
        print(f'Scores: {output["output"]}')

    if mode == "senteval" or mode == "all":
        # test usign senteval
        results = trainer.run_senteval()
        print(f"========== SENTEVAL RESULTS ==========\n{results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sent_encoder_model", type=str, required=True, help="Sentence encoder model to use: baseline, unilstm, bilstm, or bilstmmax")
    parser.add_argument("--mode", type=str, default="all", help="Mode to run: train, test, infer, senteval, or all; default is all")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training from checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer to use: adam, sgd, ; default is adam")

    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs to train for, default 20")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42")

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
