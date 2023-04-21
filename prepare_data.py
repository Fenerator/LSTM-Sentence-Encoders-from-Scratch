import torchtext
from pathlib import Path


def prepare_data(data_path: str, batch_size: int) -> tuple:
    """_summary_

    Args:
        data_path (str):
        batch_size (int):

    Returns:
        tuple: 3 dataloaders and vocab
    """
    data_path = Path(data_path)

    # define preprocessing pipeline
    tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en_core_web_sm")
    TEXT = torchtext.data.Field(tokenize=tokenizer, lower=True, batch_first=True, include_lengths=True)
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)  # unk_token=None to prevent unk token in labels

    # get data
    print("Preparing SNLI data...")
    train, val, test = torchtext.datasets.SNLI.splits(text_field=TEXT, label_field=LABEL, root=data_path)

    # build vocabulary
    print("Building vocabulary...")
    TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(cache=data_path / "GloVe/"))
    LABEL.build_vocab(train)

    # create dataloaders
    train_dl, val_dl, test_dl = torchtext.data.BucketIterator.splits((train, val, test), batch_size=batch_size)

    return train_dl, val_dl, test_dl, TEXT.vocab
