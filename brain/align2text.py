import textgrid
import sys
from funcs import text2fmri
import pickle


def main(argv):
    PATH = argv[1]
    SENT_N = argv[2]
    for lan in ["EN", "FR", "CN"]:
        aligns = []
        for i in range(1, 10):
            tg = textgrid.TextGrid.fromFILE(f"{PATH}/annotation/{lan}/lpp{lan}_section{i}.TextGrid")
            aligns.append(text2fmri(tg, SENT_N))
        with open(f"data/{lan}_aligned_words.pickle", "wb") as p:
            pickle.dump(aligns, p)


if __name__ == "__main__":
    main(sys.argv)
