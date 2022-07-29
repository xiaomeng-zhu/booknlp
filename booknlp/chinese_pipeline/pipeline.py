import re
import time
import numpy as np
import pandas as pd
import opencc
from hanlp_restful import HanLPClient
import poetry_detector
from ner_honorifics import *

def client_set_up():
    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    return HanLP

def text_file_to_string_and_simplify(file_path):
    """
    Reads a file, converts traditional Chinese into simplified, stores it in a file, and outputs
    the simplified contents of the file as a string.
    Input: path to the text file to read, string
    Output: string containing simplified Chinese
    """
    converter = opencc.OpenCC("t2s.json")
    with open(file_path, "r") as f:
        text_string = f.read()
        simplified_string = converter.convert(text_string)
        print(simplified_string[:2000])
    with open(file_path[:-4] + "_simplified.txt", "w") as f:
        f.write(simplified_string)
    return simplified_string

def strip_header_footer(doc):
    """
    Strips the Project Gutenberg header and footer from a file.
    Input: the text as a string
    Output: clean version of the text
    """
    header_idx = -1
    footer_idx = len(doc)
    try:
        header_idx = re.search("START OF (THE|THIS) PROJECT GUTENBERG EBOOK", doc).end() # index of the last character in the header identification string
    except AttributeError:
        print("No Project Gutenberg header found.")

    # res = doc
    # if header_idx != -1: # if there is a header
    res = doc[(header_idx+1):] # if header_idx == -1, there is no header, so res is doc; otherwise, res is the slice without header

    try:
        footer_idx = re.search("End of the Project Gutenberg EBook of", res).start() # index of the first character in the footer identification string
    except AttributeError:
        print("No Project Gutenberg footer found.")
    
    res = res[:footer_idx]
    res = re.sub("\n", "", res)
    res = re.sub(re.escape("*"), "", res) # for some reason, this doesn't work when escaped * is added to punctuation list
    return res

def split_text(text_string, max_length):
    """
    Splits the text by sentences into sections around max_length (exceeds max_length to split at sentence boundary)
    Input: the text as a string, the maximum length of each section
    Output: sections as a list of strings, starting character index of each section
    """
    # split by sentences and into sections around max_length (exceeds max_length to split at sentence boundary)
    # return splitted sections as a list and the starting character index of every section
    punc = ["。", "﹗", "！", "？", "．", ". ", "\u3000", "! ", "? ", "……"]
    #quote_punc = ["」", "”", "』", "`", "'"]
    quote_punc = ["」","”", "'", "』"]
    sections = []
    sections_indices = []
    finished = False
    
    section_start = 0
    section_end = max_length

    if len(text_string) < max_length:
        sections.append(text_string)
        sections_indices.append(0)
        return sections, sections_indices

    while not finished:
        if len(text_string[section_start:]) <= (max_length / 2):
            # if what remains is less than half of max_length, append the string to the last section
            sections[-1] += text_string[section_start:]
            # section indices list is not updated in this case
            finished = True
        elif (len(text_string[section_start:]) > (max_length / 2)) and (len(text_string[section_start:]) <= max_length):
            # if what remains is long enough to count as a section, append to sections list
            sections.append(text_string[section_start:])
            sections_indices.append(section_start)
            finished = True
        else:
            # if text_string[section_end] in punc:
            #     if text_string[section_end + 1] in quote_punc:
            if text_string[section_end-1] in punc:
                if text_string[section_end] in quote_punc: # extra check for quotation mark
                    section_end += 1
                sections.append(text_string[section_start:section_end])
                sections_indices.append(section_start)
                # sections.append(text_string[section_start:section_end + 1])
                section_start = section_end
                # section_start = section_end + 1
                section_end = section_start + max_length
            else:
                section_end = section_end + 1

    return sections, sections_indices

def preprocess(text_file, text_title):
    """
    Input: path to the text, name of the text
    Output: cleaned sections
    """
    HanLP = client_set_up()

    text_string = text_file_to_string_and_simplify(text_file)
    clean_string = strip_header_footer(text_string)

    # temporarily disabled because of name 'writer' is not defined errors
    # poetry_detector.extract_and_output_poetry(clean_string, text_title)

    sections, indices = split_text(clean_string, 14500) # maximum character for HanLP is 15000

    return sections

def tokenize_and_pos(client, sections, text_title):
    """
    Produces a file containing the sentence ID, token ID, and POS tag of each token
    Input: HanLP client, previously split sections, name of the text
    Output: all tokens as a list of lists of token strings, separated by sentence
    """
    all_toks = [] # list of lists of token
    all_poss = []
    all_toks_indices = []
    all_sents_indices = []
    sent_offset = 0
    token_offset = 0

    for section in sections:
        toks_pos_dict = client(section, tasks='pos/pku')
        toks = toks_pos_dict["tok/fine"] # list of lists
        
        tok_len_list = [len(tok_list) for tok_list in toks]
        num_toks = sum(tok_len_list)
        num_sents = len(toks)

        toks_indices = [idx + token_offset for idx in list(range(num_toks))]
        sents_indices = [[idx + sent_offset] * length for idx, length in enumerate(tok_len_list)]
        
        poss = toks_pos_dict["pos/pku"]
        
        all_toks += toks
        all_poss += poss
        all_toks_indices.append(toks_indices)
        all_sents_indices += sents_indices

        token_offset += num_toks # number of tokens in the section
        sent_offset += num_sents # number of sentences in the section

    toks_pos_df = pd.DataFrame()
    toks_pos_df["sentence_id"] = list(np.concatenate(all_sents_indices).flat)
    toks_pos_df["token_id"] = list(np.concatenate(all_toks_indices).flat)
    tokens_flattend = list(np.concatenate(all_toks).flat)
    toks_pos_df["token"] = tokens_flattend
    toks_pos_df["POS_tag"] = list(np.concatenate(all_poss).flat)

    toks_pos_df.to_csv("chinese_pipeline/outputs/{}_tokens.csv".format(text_title))

    return all_toks

def produce_offset(all_tokens):
    """
    
    """
    # input is a list of integers:
    res = []
    sum = 0
    for toks in all_tokens:
        res.append(sum)
        sum += len(toks)
    return res

def ner(client, all_tokens, text_title):
    ner_df = pd.DataFrame()
    all_ners = client(tokens=all_tokens, tasks="ner/msra")["ner/msra"] # list of lists
    offsets = produce_offset(all_tokens) # list is the same length as number of sentences
    all_ners_converted = []

    for sent_idx, sent_ners in enumerate(all_ners):
        offset = offsets[sent_idx]
        for ner in sent_ners:
            ner_converted = ner_match_convert(ner, all_tokens[sent_idx], offset)
            all_ners_converted.append(ner_converted)

    ner_df = pd.DataFrame(all_ners_converted, columns=["text", "cat", "start_token", "end_token"])
    ner_char_df = ner_df[ner_df.cat == 'PERSON'][["text", "start_token", "end_token"]]
    ner_df.to_csv("chinese_pipeline/outputs/{}_entities.csv".format(text_title))
    ner_char_df.to_csv("chinese_pipeline/outputs/{}_named_characters.csv".format(text_title))

    return all_ners_converted

def get_unique_names(all_ners_converted):
    ner_df = pd.DataFrame(all_ners_converted, columns=["text", "cat", "start_token", "end_token"])
    unique_names = list(ner_df[ner_df.cat == 'PERSON']["text"].unique())
    return unique_names



# def coref(client, file_path, text_title):
#     text_string = text_file_to_string(file_path)

#     # get a list of indices on how to split the string into 1000 characters
#     coref_sections, coref_section_offsets = get_coref_section_indices(text_string)
#     for section in coref_sections:
#         coref_clusters = client.coreference_resolution(section)["clusters"]


def process(text_file, text_title):
    sections = preprocess(text_file, text_title)

    HanLP = client_set_up()
    time0 = time.perf_counter()

    tokens = tokenize_and_pos(HanLP, sections, text_title)
    ners = ner(HanLP, tokens, text_title)
    unique_names = get_unique_names(ners)
    # np.savetxt("chinese_pipeline/coref_training_data/{}_named_characters.txt".format(text_title), unique_names)
    with open("chinese_pipeline/coref_training_data/{}_unique_names.txt".format(text_title), "w") as writer:
        writer.write("\n".join(unique_names))

    # TODO: 
    # coref
    # corefs = get_coref_training(sections, [0, 646, 1252, 1806], unique_names)
    # print(corefs)
    # coref_cluster
    time1 = time.perf_counter()
    print(time1-time0)

    return tokens


if __name__ == "__main__":
    # sections = preprocess("data/fengshou_excerpt.txt")
    file_paths = [
        "chinese_evaluation/examples/with_poetry/jinpingmei_chapter1.txt",
        "chinese_evaluation/examples/with_poetry/niehaihua_excerpt.txt",
        "chinese_evaluation/examples/lu_xun/ah_q_chapter12.txt",
        "chinese_evaluation/examples/linglijiguang_chapter1.txt"
    ]
    file_path = file_paths[3]
    text_title = "linglijiguang"
    res = process(file_path, text_title)

