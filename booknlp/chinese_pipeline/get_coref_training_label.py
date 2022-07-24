import pandas as pd

def read_features(text_title):
    features = pd.read_csv("coref_training_data/{}_features.csv".format(text_title))
    return features

def read_gold_standard(text_title):
    gold_standard = pd.read_csv("")
    pass
