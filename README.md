# advanced-techniques-computational-semantics

## Preparations

download glove vectors:

```bash
mkdir data/GloVe
curl -Lo data/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip data/GloVe/glove.840B.300d.zip -d data/GloVe/
rm data/GloVe/glove.840B.300d.zip
```

install torchtext v 0.6.0:

 ```bash
 pip install torchtext==0.6.0
 ```

clone infersent repo:

```bash
git clone https://github.com/facebookresearch/SentEval.git 
cd SentEval/
python setup.py install    
cd data/downstream/
```

CHANGE: download data for SentEval using command from [this repo](https://github.com/princeton-nlp/SimCSE/blob/main/SentEval/data/downstream/download_dataset.sh), original script seems deprecated.

```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar
tar xvf senteval.tar
```
