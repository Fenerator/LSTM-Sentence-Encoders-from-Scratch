import tgt
import sys
from funcs import text2fmri, extract_sent_list_from_tree_file
import pickle
import textgrid
from pathlib import Path
import pandas as pd


def main():
    # SETTINGS
    PATH = Path("/project/gpuuva021/shared/FMRI-Data")

    # get sentence information from tree files
    sentences = {
        "EN": extract_sent_list_from_tree_file(PATH / "annotation/EN/lppEN_tree.csv"),
        # "FR": extract_sent_list_from_tree_file(PATH / 'annotation/FR/lppFR_tree.csv'),
        # "CN": extract_sent_list_from_tree_file(PATH / 'annotation/CN/lppCN_tree.csv')
    }

    OUTPATH = PATH / "text_data"
    OUTPATH.mkdir(parents=True, exist_ok=True)
    SENT_N = 2  # TODO

    for language in sentences.keys():
        scans = []

        # read word informations
        word_df = pd.read_csv(
            PATH / f"annotation/{language}/lpp{language}_word_information.csv"
        )

        # keep only relevant columns
        word_df = word_df[["word", "onset", "offset", "section"]]

        # replace the last word with the word + #
        for i, sent in enumerate(sentences[language]):
            sentences[language][i][-1] = sent[-1] + "#"

        # flatten the list of lists of words into a list of words
        words = [item for sublist in sentences[language] for item in sublist]

        # extract as lists
        onsets, offsets, sections = (
            word_df["onset"].tolist(),
            word_df["offset"].tolist(),
            word_df["section"].tolist(),
        )

        # build-up the data (list of dicts with keys sentence, onset, offset, section)
        data, sentence, temp_onsets, temp_offsets, temp_sections = [], "", [], [], []

        for i, word in enumerate(words):
            sentence = sentence + word + " "
            temp_offsets.append(offsets[i])
            temp_onsets.append(onsets[i])
            temp_sections.append(sections[i])

            if word[-1] == "#":
                data.append(
                    {
                        "sentence": sentence[:-2] + ".",
                        "onset": temp_onsets[0],
                        "offset": temp_offsets[-1],
                        "section": sections[0],
                    }
                )

                # reset
                sentence, temp_onsets, temp_offsets, temp_sections = (
                    "",
                    [],
                    [],
                    [],
                )

        print(f"Language: {language} has {len(data)} sentences")
        print(data[:2])

        # split by section
        ...
        with open(f"{OUTPATH}/{lan}_aligned_words.pickle", "wb") as p:
            pickle.dump(words, p)
        with open(f"{OUTPATH}/{lan}_aligned_slices.pickle", "wb") as p:
            pickle.dump(scans, p)


if __name__ == "__main__":
    main()
