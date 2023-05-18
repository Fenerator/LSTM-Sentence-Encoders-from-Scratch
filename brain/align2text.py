import tgt
import sys
from funcs import text2fmri
import pickle
import textgrid


def main():
    # SETTINGS
    PATH = "/project/gpuuva021/shared/FMRI-Data"
    OUTPATH = f"{PATH}/text_data"
    SENT_N = 2

    for lan in ["EN", "FR", "CN"]:
        words = []
        scans = []
        for i in range(1, 10):
            tg = textgrid.TextGrid.fromFile(
                f"{PATH}/annotation/{lan}/lpp{lan}_section{i}.TextGrid"
            )
            word, sections = text2fmri(tg, SENT_N)

            print(f"{lan}_Section {i} has {len(word)} chunks")
            # print(f"{word[:5]}")

            words.append(word)
            scans.append(sections)
        with open(f"{OUTPATH}/{lan}_aligned_words.pickle", "wb") as p:
            pickle.dump(words, p)
        with open(f"{OUTPATH}/{lan}_aligned_slices.pickle", "wb") as p:
            pickle.dump(scans, p)


if __name__ == "__main__":
    main()
