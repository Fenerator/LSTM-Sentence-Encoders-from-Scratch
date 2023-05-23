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
        sentences.append(words)  # list of list of words

    print(f"Errors: {counter}")
    return sentences


def get_section_data(word_df, section):
    df_by_section = word_df[word_df["section"] == section]

    words, onsets, offsets, sections = (
        df_by_section["word"].tolist(),
        df_by_section["onset"].tolist(),
        df_by_section["offset"].tolist(),
        df_by_section["section"].tolist(),
    )

    # create list of dicts
    section_data = []
    sentence = ""
    temp_onsets, temp_offsets, temp_sections = [], [], []
    for i, word in enumerate(words):
        sentence = sentence + word + " "
        temp_offsets.append(offsets[i])
        temp_onsets.append(onsets[i])
        temp_sections.append(sections[i])

        if word[-1] == "#":
            section_data.append(
                {
                    "sentence": sentence[:-2] + ".",
                    "onset": temp_onsets[0],
                    "offset": temp_offsets[-1],
                    "section": sections[0],
                }
            )
            sentence, temp_onsets, temp_offsets, temp_sections = (
                "",
                [],
                [],
                [],
            )  # reset

    return section_data


def align_trees_with_csv_annotations(sentences, language, word_df, chunck_size=1):
    # replace the last word with the word + #
    for i, sent in enumerate(sentences):
        sentences[i][-1] = sent[-1] + "#"

    # flatten the list of lists of words into a list of words
    words = [item for sublist in sentences for item in sublist]

    # integrate words back into the dataframe
    word_df["word"] = words

    # keep only relevant columns of the dataframe
    word_df = word_df[["word", "onset", "offset", "section"]]

    # get the number of unique sections
    possible_sections = word_df["section"].unique()

    # extract as lists, for each section individually
    data = []
    for section in possible_sections:
        section_data = get_section_data(word_df, section)

        # concatenate the sentences into chunks of size chunck_size

        # add the section's data to the list of all data
        data.append(section_data)

        print(f"{language} Section {section} has {len(section_data)} chunks")

    return data


def text2fmri(textgrid, sent_n, delay=5, lan=None, sentences=None):
    # OLD
    scan_idx = []
    chunks = []
    textgrid = textgrid.tiers
    chunk = ""
    sent_i = 1
    idx_start = int(delay / 2)

    if lan == "EN":
        for interval in textgrid[0].intervals[1:]:
            # print(interval.mark)
            # print(interval.__dict__)
            # different marks depending on the language (EN, CN, FR)
            if (
                interval.mark == "#" or interval.mark == "sil"
            ):  # or interval.mark == "":
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
