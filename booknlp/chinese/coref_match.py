import pandas as pd
import json

# step 1: import csv
fs_m = pd.read_csv("annotation/miranda_coref_fs.csv")
hly_m = pd.read_csv("annotation/miranda_coref_hly.csv")
fs_k = pd.read_csv("annotation/kiara_coref_fs.csv")
hly_k = pd.read_csv("annotation/kiara_coref_hly.csv")

# step 2: convert csv into json
def csv_to_json(pd_df):
    json_dict = {}
    json_dict["type"] = "clusters"

    pd_df["start-end"] = pd_df["start_idx"].astype(str) + "-" + pd_df["end_idx"].astype(str)

    # json_dict_cluster = {} # keys being cluster idx strings and values being (start_idx, end_idx) tuples
    json_dict_cluster = pd_df.groupby('Cluster Idx')["start-end"].apply(list).to_dict()
    # print(json_dict_cluster)
    json_dict["clusters"] = json_dict_cluster
    return json_dict

json_hly_m = csv_to_json(hly_m)
json_hly_k = csv_to_json(hly_k)
json_fs_m = csv_to_json(fs_m)
json_fs_k = csv_to_json(fs_k)


# step 3: output json files
with open("coref_metrics/json_hly_m.json", "w") as outfile:
    json.dump(json_hly_m, outfile)

with open("coref_metrics/json_hly_k.json", "w") as outfile:
    json.dump(json_hly_k, outfile)

with open("coref_metrics/json_fs_m.json", "w") as outfile:
    json.dump(json_fs_m, outfile)

with open("coref_metrics/json_fs_k.json", "w") as outfile:
    json.dump(json_fs_k, outfile)



