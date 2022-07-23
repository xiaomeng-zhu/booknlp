import pandas as pd
import json
import opencc
from collections import Counter
from numpy.linalg import norm
from numpy import dot
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from coref_preprocessing import get_all_coref_lists

pronouns = ["他", "他们", "她", "她们", "祂", "你", "你们", "您", "我", "我们", "汝", "吾", "俺", "俺们", "自己", "大家", "咱", "咱们", "朕", "尔"]

def get_bert_embeddings(token_string):
    # helper function that returns the bert embedding from the given string
    # load chinese roberta models
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

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

def get_top_mention_embedding_pair(top_mention_pair):
    top_mention_A, top_mention_B = top_mention_pair

    a = get_bert_embeddings(top_mention_A)
    # print(a.shape)
    b = get_bert_embeddings(top_mention_B)
    # print(b.shape)

    return a, b

def top_mention_cosine_similarity(embedding_pair):
    a, b = embedding_pair
    
    cos_sim = dot(a, b.T)/(norm(a)*norm(b))

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

def min_index_distance(cluster_pair):
    # clusterA, clusterB = cluster_pair
    # end_max_clusterA = max([end_idx for start_idx, end_idx, mention in clusterA])
    # start_min_clusterB = min([start_idx for start_idx, end_idx, mention in clusterB])

    # start_min_clusterA = min([start_idx for start_idx, end_idx, mention in clusterA])
    # end_max_clusterB = max([end_idx for start_idx, end_idx, mention in clusterB])

    # return min(end_max_clusterA-start_min_clusterB, )
    pass

def get_feature_vector(cluster_pair):
    # top mention strings from both clusters
    top_mention_pair = get_top_mention_pair(cluster_pair)

    embedding_pair = get_top_mention_embedding_pair(top_mention_pair) # (1, 768) each
    cosine_similarity = top_mention_cosine_similarity(embedding_pair) # (1, 1)

    overlap = [[top_mention_character_overlap(top_mention_pair)]] # (1, 1)

    size_diff = [[get_size_diff(cluster_pair)]] # (1, 1)

    embedding_vector = np.hstack(embedding_pair) # (1, 1536)

    vector = np.hstack((embedding_vector, cosine_similarity, overlap, size_diff))
    print(vector.shape)
    return vector

def get_all_cluster_pair_combinations(text_title):
    pass
    # need to call get_all_coref_lists from coref_preprocessing.py

if __name__ == "__main__":
    # get_bert_embeddings("西门庆")
    cluster = [(3832, 3834, '西门庆'), (3983, 3983, '哥'), (4031, 4033, '西门庆'), (4044, 4044, '你'), (4048, 4048, '你'), (4085, 4087, '西门庆'), (4193, 4195, '西门庆'), (4210, 4210, '哥'), (4244, 4246, '西门庆'), (4277, 4279, '西门庆'), (4388, 4388, '哥'), (4447, 4449, '西门庆'), (4472, 4472, '我'), (4487, 4487, '我'), (4510, 4510, '哥'), (4582, 4584, '西门庆'), (4620, 4622, '你西门'), (4621, 4622, '西门'), (4676, 4678, '西门庆'), (4726, 4726, '哥'), (4747, 4747, '哥'), (4760, 4762, '西门庆'), (4812, 4814, '西门庆'), (4874, 4876, '西门庆'), (4919, 4921, '西门庆'), (4940, 4940, '俺'), (4945, 4945, '俺'), (5049, 5051, '西门庆'), (5160, 5160, '你'), (5165, 5165, '你'), (5174, 5174, '你')]
    cluster_2 = [(2277, 2280, '这西门庆'), (2315, 2315, '他'), (2321, 2321, '他'), (2353, 2353, '他'), (2356, 2356, '他'), (2365, 2365, '他'), (2373, 2378, '这西门大官人'), (2502, 2504, '西门庆'), (2511, 2511, '他'), (2548, 2550, '西门庆'), (2682, 2684, '西门庆'), (2860, 2862, '西门庆'), (3056, 3058, '西门庆')]
    
    get_feature_vector((cluster, cluster_2))
    