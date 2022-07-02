from model_test import *
import opencc
from ner_match import ner_match_percentage, ner_df_to_list, ner_metrics

model_list = [
    "hanlp",
    # "lac",
    # "thulac",
    # "jiagu",
]

with open("annotation_50.txt", "r") as f:
    batch_1 = f.readlines()
    batch_1 = [sent.rstrip('\n') for sent in batch_1]

with open("annotation_50_batch2.txt", "r") as f:
    batch_2 = f.readlines()
    batch_2 = [sent.rstrip('\n') for sent in batch_2]

batch_all = batch_1 + batch_2
converter = opencc.OpenCC('t2s.json')
batch_all = [converter.convert(sentence) for sentence in batch_all]

# ner_standards = pd.read_csv("annotation/ner_standard_final.csv")
ner_standards = pd.read_csv("annotation/ner_standards_final_v2.csv")
standards_list = ner_df_to_list(ner_standards)
standards_list_simplified = [(sent_idx, start_idx, end_idx, ner_type, converter.convert(tok))for sent_idx, start_idx, end_idx, ner_type, tok in standards_list]
# print(standards_list_simplified[:10])

# perc1, inhan_notinstd = ner_match_percentage(hanlp_ners_tup_list, standards_list)
# perc2, instd_notinhan = ner_match_percentage(standards_list, hanlp_ners_tup_list)
# print(perc1, perc2)

for model in model_list:
    model_res = ner_process_all(model, batch_all)

    model_res_df = pd.DataFrame(model_res, columns=["sent", "start_idx", "end_idx", "type", "string"])
    model_res_df.to_csv("model_results/{}_ner.csv".format(model))

    model_precision, model_recall, model_f1 = ner_metrics(model_res, standards_list_simplified)
    print(model, model_precision, model_recall, model_f1)

# hanlp 0.7876106194690266 0.644927536231884 0.7091633466135459
# hanlp_v2 0.8672566371681416 0.7101449275362319 0.7808764940239045 (stripping honorifics from standards)
# lac 0.7560975609756098 0.6739130434782609 0.7126436781609194
# thulac 0.5955056179775281 0.38405797101449274 0.46696035242290745
# jiagu 0.24 0.13043478260869565 0.16901408450704222
