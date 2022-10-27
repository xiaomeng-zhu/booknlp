import pandas as pd
import json
import opencc
from collections import Counter
import numpy as np
from itertools import combinations
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from coref_preprocessing import create_hanlp_client, get_names_list, split_coref_sections, get_all_coref_lists

pronouns = ["他", "他们", "她", "她们", "祂", "你", "你们", "您", "我", "我们", "汝", "吾", "俺", "俺们", "自己", "大家", "咱", "咱们", "朕", "尔"]

file_paths = [
    "chinese_evaluation/examples/with_poetry/jinpingmei_chapter1_simplified.txt",
    "chinese_evaluation/examples/with_poetry/niehaihua_excerpt_simplified.txt",
    "chinese_evaluation/examples/lu_xun/ah_q_chapter12_simplified.txt",
    "chinese_evaluation/examples/linglijiguang_chapter1_simplified.txt"
]

text_titles = ["jinpingmei", "niehaihua", "ah_q", "linglijiguang"]

jinpingmei_idx = [0, 1269, 2210, 3094, 3759]
niehaihua_idx = [0, 1318]
ahq_idx = [0, 1016, 2118, 3078]
linglijiguang_idx = [0, 1170]

offsets_list = [jinpingmei_idx, niehaihua_idx, ahq_idx, linglijiguang_idx]

def set_up_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    return tokenizer, model

def get_bert_embeddings(tokenizer, model, token_string):
    # helper function that returns the bert embedding from the given string

    input_ids = tf.constant(tokenizer.encode(token_string))[None, :]  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    embedding = last_hidden_states.numpy()
    mean_embedding = embedding.mean(axis=1)
    return mean_embedding

def get_top_mention(cluster):
    # helper function that returns the mention string that is the most frequent in the cluster and not a pronoun
    mention_string_list = [mention_string for _,_,mention_string in cluster]
    counter_dict = dict(Counter(mention_string_list))
    sorted_list = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)

    for mention, count in sorted_list:
        if mention not in pronouns:
            return mention
    return sorted_list[0][0]

def get_top_mention_pair(cluster_pair):
    clusterA, clusterB = cluster_pair

    top_mention_A = get_top_mention(clusterA)
    top_mention_B = get_top_mention(clusterB)
    
    return top_mention_A, top_mention_B

def get_top_mention_embedding_pair(tokenizer, model, top_mention_pair):
    top_mention_A, top_mention_B = top_mention_pair

    a = get_bert_embeddings(tokenizer, model, top_mention_A)
    # print(a.shape)
    b = get_bert_embeddings(tokenizer, model, top_mention_B)
    # print(b.shape)

    return a, b

def top_mention_cosine_similarity(embedding_pair):
    a, b = embedding_pair
    
    cos_sim = np.dot(a, b.T)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cos_sim

def min_edit_distance(string1, string2):
    n = len(string1)
    m = len(string2)

    matrix = [[i+j for j in range(m+1)] for i in range(n+1)]
    # print(matrix)

    for i in range(1, n+1):
        for j in range(1, m+1):
            if string1[i-1] == string2[j-1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    distance_score = matrix[n][m]
   
    return distance_score

def top_mention_character_overlap(top_mention_pair):
    top_mention_A, top_mention_B = top_mention_pair

    return min_edit_distance(top_mention_A, top_mention_B)

def get_size_diff(cluster_pair):
    clusterA, clusterB = cluster_pair
    return abs(len(clusterA) - len(clusterB))

def closest_array_items(a1, a2):
    # helper function that gets the closest pair from two arrays
    if not a1 or not a2:
        raise ValueError('Empty array')
    a1, a2  = iter(sorted(a1)), iter(sorted(a2))
    i1, i2 = a1.__next__(), a2.__next__()
    min_dif = float('inf')
    while 1:
        dif = abs(i1 - i2)
        if dif < min_dif:
             min_dif = dif
             pair = i1, i2
             if not min_dif:
                  break
        if i1 > i2:
            try:
                i2 = a2.__next__()
            except StopIteration:
                break
        else:
            try:
                i1 = a1.__next__()
            except StopIteration:
                break
    return pair

def get_min_index_distance(cluster_pair):
    clusterA, clusterB = cluster_pair
    end_clusterA = [end_idx for _,end_idx,_ in clusterA]
    start_clusterB = [start_idx for start_idx,_,_ in clusterB]
    idx1, idx2 = closest_array_items(end_clusterA, start_clusterB)
    # print(idx1, idx2)

    start_clusterA = [start_idx for start_idx,_,_ in clusterA]
    end_clusterB = [end_idx for _,end_idx,_ in clusterB]
    idx3, idx4 = closest_array_items(start_clusterA, end_clusterB)
    # print(idx3, idx4)

    return min((abs(idx1 - idx2)), (abs(idx3 - idx4)))

def get_feature_vector(tokenizer, model, cluster_pair):
    # top mention strings from both clusters
    top_mention_pair = get_top_mention_pair(cluster_pair)

    embedding_pair = get_top_mention_embedding_pair(tokenizer, model, top_mention_pair) # (1, 768) each
    embedding_vector = np.hstack(embedding_pair) # (1, 1536)

    cosine_similarity = top_mention_cosine_similarity(embedding_pair) # (1, 1)

    overlap = [[top_mention_character_overlap(top_mention_pair)]] # (1, 1)s

    size_diff = [[get_size_diff(cluster_pair)]] # (1, 1)

    min_index_distance = [[get_min_index_distance(cluster_pair)]] # (1, 1)

    vector = np.hstack((embedding_vector, cosine_similarity, overlap, size_diff, min_index_distance))
    # print(vector.shape) # (1, 1540)
    
    return vector

def filter_coref_lists_by_end_index(coref_lists):
    for cluster in coref_lists:
        # all_prev_indices = []
        to_remove = [] # list of mentions to remove
        # print(cluster)
        for idx, (start_idx, end_idx, text) in enumerate(cluster):
            if idx != 0:
                prev_start, prev_end, prev_text = cluster[idx-1]
                if prev_start >= start_idx and prev_end <= end_idx: # a, b, c
                    # remove prev
                    to_remove.append((prev_start, prev_end, prev_text))
                elif start_idx >= prev_start and end_idx <= prev_end:
                    # remove current
                    to_remove.append((start_idx, end_idx, text))
            # remove_current = False
            # if is the first mention, only append
            # if idx == 0:
            #     all_prev_indices.append((start_idx, end_idx, text))
            # else:
            #     # check through existing previous mentions for nested mentions
            #     for (existing_start, existing_end, existing_text) in all_prev_indices:
            #         if existing_start <= start_idx and existing_end >= end_idx:
            #             to_remove.append((start_idx, end_idx, text))
            #             remove_current = True
            #     if remove_current == False:
            #         all_prev_indices.append((start_idx, end_idx, text))
            # print(all_prev_indices)
        for remove_this in to_remove:
            # print(remove_this)
            cluster.remove(remove_this)
    return coref_lists
            

def get_all_cluster_pair_combinations(text_title, text_index):
    # the four functions below are imported from coref_preprocessing.py
    HanLP = create_hanlp_client()
    names_list = get_names_list(text_title)
    coref_sections = split_coref_sections(file_paths[text_index], offsets_list[text_index])
    coref_lists = get_all_coref_lists(HanLP, coref_sections, offsets_list[text_index], names_list)
    coref_lists = filter_coref_lists_by_end_index(coref_lists)
    
    coref_combinations =  list(combinations(coref_lists, 2))
    coref_comb_df = pd.DataFrame(coref_combinations)
    # coref_comb_df.to_csv("chinese_pipeline/coref_training_data/{}_all_coref_comb.csv".format(text_title))
    coref_comb_df.to_csv("chinese_pipeline/coref_training_data/{}_all_coref_comb_end_filtered.csv".format(text_title))
    return coref_combinations

def min_max_scaling(series):
    # helper function that normalizes the given series
    return (series - series.min()) / (series.max() - series.min())

def generate_feature_df(text_title, coref_pairs):
    tokenizer, model = set_up_tokenizer_and_model()
    features = [get_feature_vector(tokenizer, model, (cluster1, cluster2)) for cluster1, cluster2 in coref_pairs]
    features = np.reshape(features, (len(features), 1540))

    print(features.shape)
    features_df = pd.DataFrame(features)

    # standardize the last three fields in the row
    features_df.iloc[:, 1539] = min_max_scaling(features_df.iloc[:, 1539])
    features_df.iloc[:, 1538] = min_max_scaling(features_df.iloc[:, 1538])
    features_df.iloc[:, 1537] = min_max_scaling(features_df.iloc[:, 1537])

    features_df.to_csv("chinese_pipeline/coref_training_data/{}_features.csv".format(text_title))
    return features_df
    

if __name__ == "__main__":
    # get_bert_embeddings("西门庆")
    # cluster = [(3832, 3834, '西门庆'), (3983, 3983, '哥'), (4031, 4033, '西门庆'), (4044, 4044, '你'), (4048, 4048, '你'), (4085, 4087, '西门庆'), (4193, 4195, '西门庆'), (4210, 4210, '哥'), (4244, 4246, '西门庆'), (4277, 4279, '西门庆'), (4388, 4388, '哥'), (4447, 4449, '西门庆'), (4472, 4472, '我'), (4487, 4487, '我'), (4510, 4510, '哥'), (4582, 4584, '西门庆'), (4620, 4622, '你西门'), (4621, 4622, '西门'), (4676, 4678, '西门庆'), (4726, 4726, '哥'), (4747, 4747, '哥'), (4760, 4762, '西门庆'), (4812, 4814, '西门庆'), (4874, 4876, '西门庆'), (4919, 4921, '西门庆'), (4940, 4940, '俺'), (4945, 4945, '俺'), (5049, 5051, '西门庆'), (5160, 5160, '你'), (5165, 5165, '你'), (5174, 5174, '你')]
    # cluster_2 = [(2277, 2280, '这西门庆'), (2315, 2315, '他'), (2321, 2321, '他'), (2353, 2353, '他'), (2356, 2356, '他'), (2365, 2365, '他'), (2373, 2378, '这西门大官人'), (2502, 2504, '西门庆'), (2511, 2511, '他'), (2548, 2550, '西门庆'), (2682, 2684, '西门庆'), (2860, 2862, '西门庆'), (3056, 3058, '西门庆')]
    
    # coref_lists = [cluster, cluster_2]

    # coref_lists = [
    #     [(4891, 4899, '一个才留头的小厮儿'), (5002, 5003, '小的'), (5104, 5106, '那小厮'), (5104, 5107, '那小厮儿'), (5112, 5113, '小的'), (5150, 5150, '他'), (5198, 5200, '那小厮')], 
    #     [(5131, 5135, '大丫头玉箫'), (5132, 5133, '丫头')], \
    #     [(5170, 5172, '你家娘'), (5176, 5179, '西门大娘'), (5188, 5188, '娘')]
    # ]

    text_titles = ["jinpingmei", "niehaihua", "ah_q", "linglijiguang"]
    for i in range(len(text_titles)):
        print(i)
        coref_pairs = get_all_cluster_pair_combinations(text_titles[i], i)
        # generate_feature_df(text_titles[i], coref_pairs)

    # # some testing code
    # # test_list = [[(3101, 3105, '一个小厮儿'), (3103, 3104, '小厮'), (3169, 3169, '他'), (3171, 3171, '他')]]
    # test_list = [[(1293, 1306, "赵太爷的儿子茂才〔１２〕先生"), (1297, 1298, '儿子'), (1312, 1314, '如此公'), (1506, 1508, '茂才公')]]
    # print(filter_coref_lists_by_end_index(test_list))