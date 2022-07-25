import pandas as pd
import ast

def read_coref_pairs(text_title):
    """
    input: title of the text
    output: a list of all cluster pairs 
    each pair is of the form ([list of mentions in clusterA], [list of mentions in clusterB])
    """
    coref_pairs = pd.read_csv("chinese_pipeline/coref_training_data/{}_all_coref_comb.csv".format(text_title))

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

def hanlp_mention_to_gold_id(mention, gold_standard_mapping):
    """
    input: a mention of the form (start_idx, end_idx, mention_string) 
    and a list of gold standard mentions (cluster_id, start_idx, end_idx, mention_string)
    output: the gold standard cluster label of the mention
    (if it is not in gold standard label (e.g. wrong index or wrong string), assign it to -1)
    """
    if mention in gold_standard_mapping:
        return gold_standard_mapping[mention]
    else:
        return -1

def assign_cluster_id(cluster_pair, gold_standard_mapping):
    # given a cluster pair, return two lists of cluster ids of all the mentions in respective clusters
    clusterA, clusterB = cluster_pair
    ids_A = [hanlp_mention_to_gold_id(mention, gold_standard_mapping) for mention in clusterA]
    ids_B = [hanlp_mention_to_gold_id(mention, gold_standard_mapping) for mention in clusterB]
    return (ids_A, ids_B)
    
    
def assign_label(label_pair):
    """
    take the gold standard labels of a cluster pair and decide whether to merge or not
    1 for merge and 0 for not merge
    e.g. hanlp clustering vs. gold standard
    [5, 5, 5, 6], [-1, -1] vs. [5, 5, 5], [6], [-1, -1] ==> no merge
    [7, 7, 7, 5, 9], [7, 7] vs. [7, 7, 7, 7, 7, 7], [5], [9] ==> merge
    """
    # do we need a hard-coded rule that assigns "no merge" if all mentions in one cluster are -1??
    # e.g. ([1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 2, 2, 1, -1, 3, 3, 3, 3, 3, 3, 3, -1, 3, 3, 3, 10, 10, 10, 10, 10], [-1, -1])
    pass

def generate_all_labels(text_title, label_pairs, gold_standard_mapping):
    # given all label pairs of this text, generate all labels:
    for label_pair in label_pairs:
        assign_label(label_pair)
    # should return a series that has the same length as features

if __name__ == "__main__":
    # mapping_dict = read_gold_standard("linglijiguang")
    # mention = (34, 42, '我的父亲宋学连牧师')
    # print(hanlp_mention_to_gold_id(mention, mapping_dict)) # expected: 1

    text_title = "linglijiguang"
    coref_pairs = read_coref_pairs(text_title)
    gold_standard = read_gold_standard(text_title)
    test_coref_pair = coref_pairs[0]
    print(assign_cluster_id(test_coref_pair, gold_standard))


    

