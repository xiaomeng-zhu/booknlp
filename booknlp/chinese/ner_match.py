import pandas as pd

def ner_match_percentage(ners1, ners2):
    # ners1 and ners2 are four-place tuples of sent, start_idx, end_idx, type
    match_count = 0
    mismatch = []
    for ner in ners1:
        if ner in ners2:
            match_count += 1
        else:
            mismatch.append(ner)
    return match_count/len(ners2), mismatch

def ner_metrics(ners_model, ners_std):
    tp = 0
    fp = 0
    tn = 0 # always 0
    fn = 0
    # tp_fp = len(ners_model)
    for ner in ners_model:
        if ner in ners_std:
            tp += 1
        else:
            fp += 1
    for ner in ners_std:
        if ner not in ners_model:
            fn += 1
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
    
def ner_df_to_list(ner_df):
    # input is a pandas data frame
    ner_list = list(ner_df.itertuples(index=False, name=None))
    return ner_list

def match_idx_with_char(input_df, sent_list):
    res = []
    ner_list = list(input_df.itertuples(index=False, name=None))
    for sent, start_idx, end_idx, type in ner_list:
        string = sent_list[sent-1][start_idx: end_idx+1]
        res.append([sent, start_idx, end_idx, type, string])
    res_df = pd.DataFrame(res, columns=["sent", "start_idx", "end_idx", "type", "string"])
    return res_df


if __name__ == "__main__":
    miranda = pd.read_csv("annotation/miranda_ner.csv")
    kiara = pd.read_csv("annotation/kiara_ner.csv")
    
    m_list = ner_df_to_list(miranda)
    k_list = ner_df_to_list(kiara)

    # perc1, inm_notink = ner_match_percentage(m_list, k_list)
    # perc2, ink_notinm = ner_match_percentage(k_list, m_list)
    # print(perc1)
    # print("In Miranda's NER set but not in Kiara's: {}".format(inm_notink))
    # print(perc2)
    # print("In Kiara's NER set but not in Miranda's: {}".format(ink_notinm))

    precision1, recall1, f1_1 = ner_metrics(m_list, k_list)
    precision2, recall2, f1_2 = ner_metrics(k_list, m_list)
    print(precision1, recall1, f1_1) # 0.8840579710144928 0.8714285714285714 0.8776978417266188
    print(precision2, recall2, f1_2) # 0.8714285714285714 0.8840579710144928 0.8776978417266188

    # match indices in the annotated file with the corresponding characters in the sentences
    with open("annotation_50.txt", "r") as f:
        batch_1 = f.readlines()
        batch_1 = [sent.rstrip('\n') for sent in batch_1]

    with open("annotation_50_batch2.txt", "r") as f:
        batch_2 = f.readlines()
        batch_2 = [sent.rstrip('\n') for sent in batch_2]

    batch_all = batch_1 + batch_2
    
    standards_v1 = pd.read_csv("annotation/ner_standard_v1.csv")
    print(standards_v1.iloc[:10])
    standards_v1_with_char = match_idx_with_char(standards_v1, batch_all)
    standards_v1_with_char.to_csv("annotation/ner_standard_v1_char.csv")
