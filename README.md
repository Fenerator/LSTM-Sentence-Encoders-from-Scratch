# advanced-techniques-computational-semantics

## File Organization

`submission_name`: contains code for practical 1.

`submission_name/logs`: contains the tensorboard logs of the training runs

`submission_name/checkpoints`: would contain the checkpoints of the training runs; can be downloaded from [here](https://drive.google.com/).

`submission_name/results.ipynb`: notebook to reproduce the results of the practical.

### SentEval Location and Data

../`senteval`: in parent directory of `submission_name`, contains code for SentEval; see below for set-up steps.

../`senteval`/data: contains all the data needed. See below for set-up steps. When training a model, `SNLI` data will be stored by default in `../senteval/data/SNLI/`.

## Code Organization

`train.py`: is the main file and contains the training/validation framework.

`model.py`: contains the model definitions.

`prepare_data.py`: contains the utility functions to load and preprocess the SNLI dataset.

## Set-up Steps

1. activate a virtual environment and install dependencies from `requirements.txt` in the virtual environment; I used python 3.8.15.
2. install infersent as follows:
   clone infersent repo:

   ```bash
    cd ..
    git clone https://github.com/facebookresearch/SentEval.git 
    cd SentEval/
    python setup.py install    
    cd data/downstream/
    ```

3. Download data for SentEval using command from [this repo](https://github.com/princeton-nlp/SimCSE/blob/main/SentEval/data/downstream/download_dataset.sh), original script seems deprecated.

    ```bash
    wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar
    tar xvf senteval.tar
    ```

4. Download glove vectors:

    ```bash
    cd ../
    mkdir GloVe
    cd GloVe/
    curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    rm glove.840B.300d.zip
    ```

5. download spacy model

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Example Commands

for all commands see `train.py`'s main function.

### Training a Model

from scratch:

```bash
python train.py --mode train --optimizer adam --sent_encoder_model bilstmmax
```

from a checkpoint: add flag `resume_training`, this uses the latest checkpoint of the model in the `checkpoints` directory.

for more options, see argparse help: `python train.py -h`

### Test a Model

```bash
python train.py --mode test --optimizer adam --sent_encoder_model bilstmmax
```
