import torchtext


def prepare_data(data_path: str, batch_sizes: tuple) -> tuple:
    """_summary_

    Args:
        data_path (str): _description_
        batch_sizes (tuple): train, val, test batch sizes

    Returns:
        tuple: 3 dataloaders and vocab
    """

    # define preprocessing pipeline
    tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en_core_web_sm")
    TEXT = torchtext.data.Field(tokenize=tokenizer, lower=True, batch_first=True, include_lengths=True)
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)  # unk_token=None to prevent unk token in labels

    # get data
    train, val, test = torchtext.datasets.SNLI.splits(text_field=TEXT, label_field=LABEL, root=data_path)

    # build vocabulary
    TEXT.build_vocab(train, vectors=torchtext.vocab.GloVe(cache=data_path + "GloVe/"))
    LABEL.build_vocab(train)

    # create dataloaders
    train_dl, val_dl, test_dl = torchtext.data.BucketIterator.splits((train, val, test), batch_sizes=batch_sizes)

    return train_dl, val_dl, test_dl, TEXT.vocab
