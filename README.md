# advanced-techniques-computational-semantics

## Preparations

install torchtext v 0.6.0:

 ```bash
 pip install torchtext==0.6.0
 ```

clone infersent repo:

```bash
cd ..
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

download glove vectors:

```bash
cd ../
mkdir GloVe
curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

goto home dir again:

```bash
cd ../../practical1/
```

was before:
curl -Lo data/GloVe/glove.840B.300d.zip <http://nlp.stanford.edu/data/glove.840B.300d.zip>
unzip data/GloVe/glove.840B.300d.zip -d data/GloVe/
rm data/GloVe/glove.840B.300d.zip
