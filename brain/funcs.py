import pandas as pd
from nltk.tree import Tree


def fmri2words(text_data, Trs, section, delay=5, window=0.2):
    chunks = []
    text = text_data[text_data["section"] == section]
    for tr in range(Trs):
        onset = tr * 2 - delay
        offset = onset + 2
        chunk_data = text[
            (text["onset"] >= onset - window) & (text["offset"] < offset + window)
        ]
        chunks.append(" ".join(list(chunk_data["word"])))
    return chunks


def extract_words_from_tree(tree):
    words = []
    if isinstance(tree, str):  # Base case: leaf node (word)
        return [tree]

    elif isinstance(tree, Tree):
        for subtree in tree:
            words.extend(extract_words_from_tree(subtree))

    # sentence = " ".join(words)
    return words


def extract_sent_list_from_tree_file(PATH):
    with open(PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    counter = 0
    for i, line in enumerate(lines):
        line = line.strip()
        try:
            tree = Tree.fromstring(line)
        except ValueError:
            try:  # remove last ')'
                tree = Tree.fromstring(line[:-1])

            except ValueError:
                counter += 1
                print(f"=== ValueError: line {i} \n {line} ===")
                continue
        words = extract_words_from_tree(tree)
        sentences.append(" ".join(words))

    print(f"Errors: {counter}")
    return sentences


def text2fmri(textgrid, sent_n, delay=5):
    scan_idx = []
    chunks = []
    textgrid = textgrid.tiers
    chunk = ""
    sent_i = 1
    idx_start = int(delay / 2)
    for interval in textgrid[0].intervals[1:]:
        # print(interval.__dict__)
        # different marks depending on the language (EN, CN, FR)
        if interval.mark == "#" or interval.mark == "sil":  # or interval.mark == "":
            chunk += "."
            if sent_i == sent_n:
                chunks.append(chunk[1:])
                idx_end = min(int((interval.maxTime + delay) / 2) + 1, 282)
                scan_idx.append(slice(idx_start, idx_end))
                sent_i = 0
                chunk = ""
                idx_start = idx_end - 1
            sent_i += 1
            continue
        chunk += " " + interval.mark
    return chunks, scan_idx
