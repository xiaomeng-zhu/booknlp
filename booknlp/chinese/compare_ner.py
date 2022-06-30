from model_test import *
import opencc
from ner_match import ner_match_percentage, ner_df_to_list, ner_metrics

# def calculate_ner_match(model, standards):


if __name__ == "__main__":
    with open("annotation_50.txt", "r") as f:
        batch_1 = f.readlines()
        batch_1 = [sent.rstrip('\n') for sent in batch_1]

    with open("annotation_50_batch2.txt", "r") as f:
        batch_2 = f.readlines()
        batch_2 = [sent.rstrip('\n') for sent in batch_2]

    batch_all = batch_1 + batch_2

    # converter = opencc.OpenCC('t2s.json')
    # batch_all = [converter.convert(sentence) for sentence in batch_all]

    hanlp_ners = hanlp_ner(batch_all)
    print(hanlp_ners[:10])
    hanlp_ners_df = pd.DataFrame(hanlp_ners, columns=["sent", "start_idx", "end_idx", "type", "string"])
    hanlp_ners_df.to_csv("model_results/hanlp_ner.csv")

    hanlp_ners_tup_list = [tuple(ner) for ner in hanlp_ners]

    ner_standards = pd.read_csv("annotation/ner_standard_final.csv")
    standards_list = ner_df_to_list(ner_standards)

    precision, recall, f1 = ner_metrics(hanlp_ners_tup_list, standards_list)
    print(precision, recall, f1)
    # 0.7876106194690266 0.644927536231884 0.7091633466135459

    # perc1, inhan_notinstd = ner_match_percentage(hanlp_ners_tup_list, standards_list)
    # perc2, instd_notinhan = ner_match_percentage(standards_list, hanlp_ners_tup_list)
    # print(perc1, perc2)
