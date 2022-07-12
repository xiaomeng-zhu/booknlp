from model_test import *

model_list = [
    # "jieba",
    # "lac",
    "hanlp",
    # "stanza", # tokenizer package version conflict
    # "jiayan", # model too large to be committed
    "jiagu",
    # "ltp", # tokenizer package version conflict
    # "thulac",
    # "pkuseg",
]

# original sentences as untokenized strings
with open("annotation/annotation_50.txt", "r") as f:
    sentences = f.readlines()
sentences = [sent.rstrip('\n') for sent in sentences]

# convert traditional chinese to simplified
converter = opencc.OpenCC('t2s.json')
sentences = [converter.convert(sent) for sent in sentences]

# gold standard tokenized result
standards_tok = pd.read_csv("annotation/standard_tokenization.csv")
standards_tok = list(standards_tok.iloc[:, 1]) # list of strings with tokens separated by / 

# convert traditional chinese to simplified
standards_tok = [converter.convert(tok_list) for tok_list in standards_tok]
standards_tok = [sent_string.split("/") for sent_string in standards_tok]

# gold standard tok and pos result
standards_tok_pos = pd.read_csv("annotation/standard_pos.csv")
standards_tok_pos = list(standards_tok_pos.iloc[:, 1])
standards_tok_pos = [converter.convert(tokpos_list) for tokpos_list in standards_tok_pos]
standards_tok_pos = parse_tok_pos(standards_tok_pos)

# convert pos in gold standard to general tagging
standards_tok_pos_converted = [convert_tokposlist_to_general("hanlp", tok_pos_list) for tok_pos_list in standards_tok_pos]

for model in model_list:
    res = process_tokenized_sents(sentences, model, standards_tok)
    res_converted = [convert_tokposlist_to_general(model, tok_pos_list) for tok_pos_list in res]

    # code for conversion between jiagu and pkuseg
    # direction = ""
    # if model == "jiagu":
        # if convert_from == "jiagu": # convert jiagu pos to pku
        #     res = list(map(convert_pos_jiagupos_pku, res))
        #     direction = "_model2pku"
        # else: # convert gold standard to jiagu pos
        #     standards_tok_pos = list(map(convert_pos_pku_jiagupos, standards_tok_pos))
        #     direction = "_standard2jiagu"

    # total_mismatch, average_mismatch, mismatch_list = pos_mismatch(standards_tok_pos, res)
    total_mismatch, average_mismatch, mismatch_list = pos_mismatch(standards_tok_pos_converted, res_converted)
    
    print(model, total_mismatch, average_mismatch)

    # output mismatch tokens
    # file_name = "{}_pos_mismatch{}.csv".format(model, direction)
    # mismatch_df = pd.DataFrame(mismatch_list, 
    #         columns=["sent_idx", "tok", "standard", model]).to_csv(file_name)
    

# ===== did not convert all to simplified =====
# average percent mismatch when convert jiagupos to pku
# hanlp 253 0.16073418054397515
# jiagu 428 0.2694341499643043

# average percent mismatch when convert gold standard pos in pku to jiagupos
# hanlp 253 0.16073418054397515
# jiagu 458 0.28878558958873424
# =============================================

# average percent mismatch when convert all to general
# hanlp 212 0.13179229570280918
# jiagu 297 0.18693994145427223
