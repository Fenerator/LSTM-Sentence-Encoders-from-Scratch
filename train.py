from dataset import Custom_Dataset

# prepare data
ds = Custom_Dataset(glove_path="data/GloVe/glove.840B.300d.txt", batch_size=1)  # access split dl like this: ds.train_dl, ds.val_dl, ds.test_dl, ds.vocab


# training
# encode premise and hypothesis with GloVE embeddings

# encode premise and hypothesis with LSTM

# classify premise and hypothesis
# all in all 4 models


# evaluation
