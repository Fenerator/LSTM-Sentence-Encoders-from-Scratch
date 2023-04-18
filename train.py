from dataset import Custom_Dataset

# prepare data
ds = Custom_Dataset(glove_path="data/GloVe/glove.840B.300d.txt", batch_size=1)  # access split dl like this: ds.train_dl, ds.val_dl, ds.test_dl

# acces first batch of data
print(next(iter(ds.test_dl["premise"])))

# training
# all in all 4 models


# evaluation
