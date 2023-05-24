import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle, os
import numpy as np

# SETTINGS
PATH = "/project/gpuuva021/shared/FMRI-Data"
OUTPATH = f"{PATH}/aligned"
SENT_N = [2, 1]  # chunksize: nr. of sentences (of the same section)
BATCH_SIZE = 32

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
model = model.to(device)

os.makedirs(OUTPATH, exist_ok=True)

for sent_n in SENT_N:
    for lang in ["EN", "FR", "CN"]:
        # for lang in ["EN"]:
        print(f"Encoding Language: {lang}, using chunk size: {sent_n}")

        # read the aligned words
        with open(
            f"{PATH}/text_data/{lang}_chunk_data_chunk_size_{sent_n}.pickle", "rb"
        ) as f:
            aligned_words = pickle.load(
                f
            )  # each section contains list of chunks which is a dict with keys sentence, onset, offset, section

        # store encodings of each chunk
        hidden_states_per_section = []
        for i, section in enumerate(aligned_words):
            print(f"Encoding Sec {i}, len(section): {len(section)}")

            assert len(section) > 0, f"Section {i} has 0 chunks \n {section}"

            # get the sentences of the section
            section_sentences = [chunk["sentences"].strip() for chunk in section]

            section_batches = []
            for i in range(0, len(section_sentences), BATCH_SIZE):
                try:
                    section_batches.append(section_sentences[i : i + BATCH_SIZE])
                except IndexError:
                    section_batches.append(section_sentences[i:])  # Rest

            print(f"Nr. of batches: {len(section_batches)}")
            print(f"batch 0: {section_batches[0]}")

            # encode the section
            section_h_s = []
            for batch in section_batches:
                print(f"Encoding batch: {batch}")
                try:
                    encoded_input = tokenizer(
                        batch, return_tensors="pt", padding=True
                    ).to(device)
                except IndexError:
                    print(f"=== IndexError: section {i} \n {section} ===")

                with torch.no_grad():
                    output = model(**encoded_input, output_hidden_states=True)

                batch_h_s = output.hidden_states

                print(f"last hidden state shape: {batch_h_s[-1].shape}")

                section_h_s.append(batch_h_s)

            hidden_states_per_section.append(list(section_h_s))

        # save the hidden states in a list of tensors (sections, batches) (shape: batch_size, sequence_length, hidden_size)
        print(
            f"Saving hidden states of to {OUTPATH}/{lang}_hidden_states_chunk_size_{sent_n}.pickle"
        )
        with open(
            f"{OUTPATH}/{lang}_hidden_states_chunk_size_{sent_n}.pickle", "wb"
        ) as f:
            pickle.dump(hidden_states_per_section, f)
