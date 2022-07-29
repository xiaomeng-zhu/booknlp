from venv import create
import pandas as pd
import numpy as np
import opencc
from pipeline import get_unique_names, split_text

def create_hanlp_client():
    from hanlp_restful import HanLPClient
    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    return HanLP

def split_coref_sections(file_path, index_list):
    # given a list of indices [0, a, b, c], split the string into sections of [0: a-1], [a: b-1], [b: c-1], [c:]
    with open(file_path, "r") as f:
        texts = f.read()
    char_idx = 0
    real_index_list = []
    sections = []

    for l_idx, l in enumerate(texts):
        if l.isspace() or l == "　":
            pass
        else:
            if char_idx in index_list:
                real_index_list.append(l_idx)
            char_idx += 1

    real_index_list.append(len(texts))

    sections = [texts[i:j] for i,j in zip(real_index_list[:-1], real_index_list[1:])]
        
    return sections

def convert_to_standard_char_index(text_string):
    # for coref char indexing, use a clean text string that is free from spaces and consistent with coref annotations
    clean_string = ""
    for l in text_string:
        if l.isspace() or l == "　":
            pass
        else:
            clean_string += l
    return clean_string

def is_character(cluster, names_list):
    """
    input: cluster to be examined, list of tuples; ner results, list of (string, tag, start_idx, end_idx)
    output: bool depending on whether or not the string follows character filtering rules
    """
    pronouns = ["他", "她", "祂", "你", "我", "汝", "俺", "自己", "大家", "咱"]
    other_indicators = ["人", "者", "神", "仙", "徒", "群", "位", "士"]
    honorifics = pd.read_csv("chinese_evaluation/annotation/honorifics.csv")
    honorifics_list = list(honorifics["simplified"])

    for _,_,mention in cluster:
        for p in pronouns:
            if p in mention: # if the mention string contains a pronoun
                return True
        for i in other_indicators:
            if i in mention: # if the mention string contains other indicators
                return True
        for h in honorifics_list:
            if h in mention: # if the mention string contains honorifics
                return True
        for n in names_list:
            if n in mention: # if the mention string contains names in the unique names list
                return True
    return False

def get_coref_section_indices(text_string):
    # given return a list of character indices that are used to separate sections for coref
    clean_string = convert_to_standard_char_index(text_string)
    
    # standard coref character counting uses clean string that is free from all spaces
    coref_sections, coref_section_indices = split_text(clean_string, 1000)
    return coref_sections, coref_section_indices

def get_coref_list_from_section(client, section, offset, names_list):
    corefs = client.coreference_resolution(section)
    
    coref_clusters = corefs["clusters"]
    coref_tokens = corefs["tokens"]

    char_idx = 0
    char_idx_list = [] # list of the starting character index of each token, should be the same length as coref_tokens

    for tok in coref_tokens:
        char_idx_list.append(char_idx)
        char_idx += len(tok)
    
    clusters_list = []
    for clust_list in coref_clusters:
        this_cluster = []
        for mention_string, start_tok_idx, end_tok_idx in tuple(clust_list):

            # convert token index to character index and add offset
            start_char_idx = char_idx_list[start_tok_idx] + offset
            end_char_idx = char_idx_list[end_tok_idx] - 1 + offset

            mention = (start_char_idx, end_char_idx, mention_string)
            this_cluster.append(mention)
        if is_character(this_cluster, names_list): # filter out non-character clusters
            clusters_list.append(this_cluster)

        # TODO: check if the following is needed. 
        # if output_dict_cluster[cluster_idx] == []:
        #     output_dict_cluster.pop(cluster_idx)

    return clusters_list

def get_names_list(text_name):
    # reading unique names from coref_training_data dir
    with open("chinese_pipeline/coref_training_data/{}_unique_names.txt".format(text_name), "r") as f:
        unique_names = f.readlines()

    return [name.strip() for name in unique_names]

def get_all_coref_lists(client, sections, offsets_list, names_list):
    # for all coref sections, get coref list and combine into a big list
    all_clusters_list = []
    for i in range(len(sections)):
        clusters_list = get_coref_list_from_section(client, sections[i], offsets_list[i], names_list)
        all_clusters_list += clusters_list

    print(all_clusters_list)
    return all_clusters_list



if __name__ == "__main__":
    text_name = "jinpingmei"
    HanLP = create_hanlp_client()
    names_list = get_names_list(text_name)

    jinpingmei_idx = [0, 1269, 2210, 3094, 3759]
    niehaihua_idx = [0, 1318]
    ahq_idx = [0, 1016, 2118, 3078]
    linglijiguang_idx = [0, 1170]

    offsets_list = [jinpingmei_idx, niehaihua_idx, ahq_idx, linglijiguang_idx]

    file_paths = [
        "chinese_evaluation/examples/with_poetry/jinpingmei_chapter1_simplified.txt",
        "chinese_evaluation/examples/with_poetry/niehaihua_excerpt_simplified.txt",
        "chinese_evaluation/examples/lu_xun/ah_q_chapter12_simplified.txt",
        "chinese_evaluation/examples/linglijiguang_chapter1_simplified.txt"
    ]

    coref_sections = split_coref_sections(file_paths[0], offsets_list[0])
    print(get_all_coref_lists(HanLP, coref_sections, offsets_list[0], names_list))
    
