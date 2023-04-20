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
cd SentEval
python setup.py install    
cd data/downstream/
```
