from model_test import *
import opencc
from ner_match import ner_match_percentage, ner_df_to_list, ner_metrics

model_list = [
    "hanlp",
    "lac",
    "thulac",
    "jiagu",
]

with open("annotation_50.txt", "r") as f:
    batch_1 = f.readlines()
    batch_1 = [sent.rstrip('\n') for sent in batch_1]

with open("annotation_50_batch2.txt", "r") as f:
    batch_2 = f.readlines()
    batch_2 = [sent.rstrip('\n') for sent in batch_2]

batch_all = batch_1 + batch_2

# convert all sentences to simplified
converter = opencc.OpenCC('t2s.json')
batch_all = [converter.convert(sentence) for sentence in batch_all]

# ner_standards = pd.read_csv("annotation/ner_standard_final.csv")
ner_standards = pd.read_csv("annotation/standard_ner.csv")
standards_list = ner_df_to_list(ner_standards)
standards_list_simplified = [(sent_idx, start_idx, end_idx, ner_type, converter.convert(tok))for sent_idx, start_idx, end_idx, ner_type, tok in standards_list]
# print(standards_list_simplified[:10])

# perc1, inhan_notinstd = ner_match_percentage(hanlp_ners_tup_list, standards_list)
# perc2, instd_notinhan = ner_match_percentage(standards_list, hanlp_ners_tup_list)
# print(perc1, perc2)

ner_sets = [
    "ner/msra",
    "ner/pku",
    "ner/ontonotes"
]
for ner_set in ner_sets:
    hanlp_res = hanlp_ner(batch_all, ner_set)
    hanlp_precision, hanlp_recall, hanlp_f1 = ner_metrics(hanlp_res, standards_list_simplified)
    print(ner_set, hanlp_precision, hanlp_recall, hanlp_f1)

# for model in model_list:
#     model_res = ner_process_all(model, batch_all)

#     model_res_df = pd.DataFrame(model_res, columns=["sent", "start_idx", "end_idx", "type", "string"])
#     model_res_df.to_csv("model_results/{}_ner.csv".format(model))

#     model_precision, model_recall, model_f1 = ner_metrics(model_res, standards_list_simplified)
#     print(model, model_precision, model_recall, model_f1)

# Result on Jun 29
# hanlp 0.7876106194690266 0.644927536231884 0.7091633466135459
# hanlp_v2 0.8672566371681416 0.7101449275362319 0.7808764940239045 (stripping honorifics from standards)
# lac 0.7560975609756098 0.6739130434782609 0.7126436781609194
# thulac 0.5955056179775281 0.38405797101449274 0.46696035242290745
# jiagu 0.24 0.13043478260869565 0.16901408450704222

# Result on Jul 1 (standard version 1)
# hanlp 0.7787610619469026 0.6470588235294118 0.7068273092369478
# lac 0.7479674796747967 0.6764705882352942 0.7104247104247104
# thulac 0.5955056179775281 0.3897058823529412 0.4711111111111111
# jiagu 0.22666666666666666 0.125 0.1611374407582938

# Result on Jul 1 (standard version 2)
# hanlp 0.8584070796460177 0.7132352941176471 0.7791164658634538
# lac 0.6829268292682927 0.6176470588235294 0.6486486486486487
# thulac 0.6067415730337079 0.39705882352941174 0.48
# jiagu 0.22666666666666666 0.125 0.1611374407582938

# Comparison of NER sets version 1

# Comparison of NER sets version 2
# ner/msra 0.8429752066115702 0.75 0.7937743190661479
# ner/pku 0.8584070796460177 0.7132352941176471 0.7791164658634538
# ner/ontonotes 0.8807339449541285 0.7058823529411765 0.7836734693877551
