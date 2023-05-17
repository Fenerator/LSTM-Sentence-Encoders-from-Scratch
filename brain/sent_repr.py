import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model = model.to(device)

PATH = "/project/gpuuva021/shared/FMRI-Data"

# for lang in ["EN", "FR", "CN"]:
for lang in ["FR"]:
    print(f"Encoding Language: {lang}")

    # read the aligned words
    with open(f"{PATH}/text_data/{lang}_aligned_words.pickle", "rb") as f:
        aligned_words = pickle.load(f)  # each section contains list of scans; list of words

    # store encodings of each chunk
    hidden_states = []
    for i, section in enumerate(aligned_words):
        print(f"Encoding Sec {i}")
        
        assert len(section) > 0, f"Section {i} has 0 chunks \n {section}"
        
        # encode the section
        try:
            encoded_input = tokenizer(section, return_tensors="pt", padding=True).to(device)
        except IndexError:
            print(f"=== IndexError: section {i} \n {section} ===")

        output = model(**encoded_input, output_hidden_states=True)

        h_s = output.hidden_states

        print(f"last hidden state shape: {h_s[-1].shape}")

        hidden_states.append(list(h_s))

    # save the hidden states in a list of tensors (shape: batch_size, sequence_length, hidden_size)
    with open(f"{PATH}/aligned/{lang}_hidden_states.pickle", "wb") as f:
        pickle.dump(hidden_states, f)
