"""
used for ner annotatioan
"""
import pandas as pd

with open("annotation_50_batch2.txt", "r") as f:
    sentences = f.readlines()

sentences = [sent.rstrip('\n') for sent in sentences]
sentences = [list(sent) for sent in sentences]
# print(sentences)
sent_len = max([len(sent) for sent in sentences])

toks_df = pd.DataFrame(sentences, index=range(51,101), columns=range(sent_len))
toks_df.to_csv("ner_batch_2.csv")