import pandas as pd
import ast
import itertools
import numpy as np
from sklearn.metrics import adjusted_rand_score
from collections import Counter
from transformers import BertTokenizer, TFBertModel
from get_coref_training_features import set_up_tokenizer_and_model, get_bert_embeddings
import pickle

def read_coref_pairs(text_title):
    """
    input: title of the text
    output: a list of all cluster pairs 
    each pair is of the form ([list of mentions in clusterA], [list of mentions in clusterB])
    """
    # coref_pairs = pd.read_csv("chinese_pipeline/coref_training_data/{}_all_coref_comb.csv".format(text_title))
    coref_pairs = pd.read_csv("chinese_pipeline/coref_training_data/{}_all_coref_comb_end_filtered.csv".format(text_title))

    coref_pairs.iloc[:, 1] = coref_pairs.iloc[:, 1].apply(lambda x: ast.literal_eval(str(x))) # every mentions list in the pair seems to be an object??
    coref_pairs.iloc[:, 2] = coref_pairs.iloc[:, 2].apply(lambda x: ast.literal_eval(str(x)))
    coref_pairs_list = coref_pairs.iloc[:, 1:3].to_records(index=False)

    return coref_pairs_list
    
def read_gold_standard(text_title):
    """
    input: title of the text
    output: a dict that has (start_idx, end_idx, mention_string) as the key 
    and cluster_id as the value
    """
    mention_to_id_dict = {}
    gold_standard_df = pd.read_csv("chinese_pipeline/coref_training_data/{}_gold.csv".format(text_title))
    gold_standard_list = list(gold_standard_df[["Cluster Idx", "string", "start_idx", "end_idx"]].to_records(index=False))
    
    for id, string, start_idx, end_idx in gold_standard_list:
        mention_to_id_dict[(start_idx, end_idx, string)] = id

    return mention_to_id_dict

def hanlp_mention_to_gold_id(mention, gold_standard_mapping, unknown_id):
    """
    input: a mention of the form (start_idx, end_idx, mention_string) 
    and a list of gold standard mentions (cluster_id, start_idx, end_idx, mention_string)
    output: the gold standard cluster label of the mention
    (if it is not in gold standard label (e.g. wrong index or wrong string), assign it to -1)
    """
    if mention in gold_standard_mapping:
        return gold_standard_mapping[mention]
    else:
        return unknown_id

def assign_cluster_id(cluster_pair, gold_standard_mapping):
    # given a cluster pair, return two lists of cluster ids of all the mentions in respective clusters
    clusterA, clusterB = cluster_pair
    ids_A = [hanlp_mention_to_gold_id(mention, gold_standard_mapping, -1) for mention in clusterA]
    ids_B = [hanlp_mention_to_gold_id(mention, gold_standard_mapping, -2) for mention in clusterB]
    return (ids_A, ids_B)

def assign_all_id(coref_pairs):
    all_ids = []
    for pair in coref_pairs:
        all_ids.append(assign_cluster_id(pair, gold_standard))
    return all_ids
    
def assign_label(id_pair):
    """
    take the gold standard labels of a cluster pair and decide whether to merge or not
    1 for merge and 0 for not merge
    e.g. hanlp clustering vs. gold standard
    [5, 5, 5, 6, -2, -2], [-1, -1] vs. [5, 5, 5], [6], [-1, -1], [-2, -2] ==> no merge score
    [5, 5, 5, 6, -2, -2, -1, -1] vs. [5, 5, 5], [6], [-1, -1], [-2, -2] ==> merge score
    compare no merge score and merge score, label whichever one is higher
    no merge if tie
    """
    id_list_A, id_list_B = id_pair
    merged_list = id_list_A + id_list_B

    most_common_item_A, most_common_count_A = Counter(id_list_A).most_common(1)[0]
    A_freq_old = most_common_count_A/len(id_list_A)

    most_common_item_B, most_common_count_B = Counter(id_list_B).most_common(1)[0]
    B_freq_old = most_common_count_B/len(id_list_B)

    merged_counter = Counter(merged_list)
    most_common_item_merged, most_common_count_merged = merged_counter.most_common(1)[0]
    A_freq_new = merged_counter[most_common_item_A]/len(merged_list)
    B_freq_new = merged_counter[most_common_item_B]/len(merged_list)
    if A_freq_new > A_freq_old:
        print(id_pair)
        return 1
    if B_freq_new > B_freq_old:
        print(id_pair)
        return 1
    return 0


def generate_all_labels(id_pairs):
    # TODO: 
    # given all label pairs of this text, generate all labels:
    all_labels = []
    for id_pair in id_pairs:
        all_labels.append(assign_label(id_pair))
    return all_labels
    # should return a series that has the same length as features
    # SCSRP work is paused here on July 29, 2022 with the decision to revisit gold standard coref annotations to include nested mentions and possessive constructions

# def get_embedding_similarity_score():
def get_pair_similarity_score(pair_wise_similarity_dict, tokenizer, model, pair):
    text1, text2 = pair
    if pair in pair_wise_similarity_dict:
        return pair_wise_similarity_dict, pair_wise_similarity_dict[pair]
    else:
        embedding1 = get_bert_embeddings(tokenizer, model, text1)
        embedding2 = get_bert_embeddings(tokenizer, model, text2)
        cos_sim = np.dot(embedding1, embedding2.T)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
        similarity_score = cos_sim.item()
        pair_wise_similarity_dict[pair] = similarity_score
        return pair_wise_similarity_dict, similarity_score
    

def calculate_average_similarity_score(pair_wise_similarity_dict, tokenizer, model, cluster):
    all_pairs_mentions = list(itertools.combinations(cluster,2))
    if len(cluster) == 1:
        return pair_wise_similarity_dict, 1
    total_sim = 0
    for mention1, mention2 in all_pairs_mentions:
        _, _, text1 = mention1
        _, _, text2 = mention2
        # print(text1, text2)
        pair = text1, text2
        pair_wise_similarity_dict, similarity_score = get_pair_similarity_score(pair_wise_similarity_dict, tokenizer, model, pair)
        total_sim += similarity_score
       
    # print(total_sim/len(all_pairs_mentions))

    return pair_wise_similarity_dict, total_sim/len(all_pairs_mentions)

def make_decision(pair_wise_similarity_dict, tokenizer, model, cluster_pair):
    cluster1, cluster2 = cluster_pair
    
    big_cluster = cluster1+cluster2
    
    pair_wise_similarity_dict, cluster1_sim = calculate_average_similarity_score(pair_wise_similarity_dict, tokenizer, model, cluster1)
    pair_wise_similarity_dict, cluster2_sim = calculate_average_similarity_score(pair_wise_similarity_dict, tokenizer, model, cluster2)
    pair_wise_similarity_dict, cluster_sim = calculate_average_similarity_score(pair_wise_similarity_dict, tokenizer, model, big_cluster)
    
    # weighted average
    # 0.81*(31/33)+0.63*(2/33)
    cluster1_len, cluster2_len = len(cluster1), len(cluster2)
    total_len = cluster1_len + cluster2_len
    weight1 = cluster1_len/total_len
    weight2 = cluster2_len/total_len

    weighted_average = cluster1_sim*weight1+cluster2_sim*weight2
    if weighted_average < cluster_sim:
        print("MERGE!", cluster1, cluster2, cluster1_sim, cluster2_sim, weighted_average)
        return pair_wise_similarity_dict, 1
    else:
        return pair_wise_similarity_dict, 0

def make_all_decision(tokenizer, model, coref_pairs):
    merge_count = 0
    total_count = 0
    pair_wise_similarity_dict = {}
    for pair in coref_pairs:
        pair_wise_similarity_dict, decision = make_decision(pair_wise_similarity_dict, tokenizer, model, pair)
        total_count += 1
        if decision == 1:
            merge_count += 1
    print(merge_count, total_count)

if __name__ == "__main__":
    # mapping_dict = read_gold_standard("linglijiguang")
    # mention = (34, 42, '我的父亲宋学连牧师')
    # print(hanlp_mention_to_gold_id(mention, mapping_dict)) # expected: 1
    tokenizer, model = set_up_tokenizer_and_model()
    text_title = "jinpingmei"
    coref_pairs = read_coref_pairs(text_title)
    # for pair in coref_pairs:
    #     make_decision(tokenizer, model, pair)
    make_all_decision(tokenizer, model, coref_pairs)
    # gold_standard = read_gold_standard(text_title)
    # id_pairs = assign_all_id(coref_pairs)
    # all_labels = generate_all_labels(id_pairs)
    # print(Counter(all_labels))
