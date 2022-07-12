from model_test import *

model_list = [
    "jieba",
    "lac", # uses modified 863
    "hanlp",
    # "stanza", # tokenizer package version conflict
    # "jiayan", # model too large to be committed
    "jiagu",
    # "ltp", # tokenizer package version conflict
    "thulac", # uses 863
    "pkuseg",
]

converter = opencc.OpenCC('t2s.json')
# read original sentences as untokenized strings
with open("annotation/annotation_50.txt", "r") as f:
    sentences = f.readlines()
sentences = [sent.rstrip('\n') for sent in sentences]

# convert traditional chinese to simplified
sentences = [converter.convert(sent) for sent in sentences]

# read gold standard tokenized result
standards_tok = pd.read_csv("annotation/standard_tokenization.csv")
standards_tok = list(standards_tok.iloc[:, 1]) # list of strings with tokens separated by / 
standards_tok = [sent_string.split("/") for sent_string in standards_tok]

# read gold standard tok and pos result
standards_tok_pos = pd.read_csv("annotation/standard_pos.csv")
standards_tok_pos = list(standards_tok_pos.iloc[:, 1]) # list of strings

# convert traditional chinese to simplified
standards_tok_pos = [converter.convert(sent) for sent in standards_tok_pos]

# get all gold standard part of speech tags
standards_pos = [] # list of lists of pos
for sent in standards_tok_pos:
    pos_list = []
    tok_pos_list = sent.split("/")
    for tok_pos in tok_pos_list:
        pos_list.append(tok_pos.split("_")[1]) # list of lists of pos
    standards_pos.append(pos_list)
standards_pos_converted = map(convert_poslist_to_general, "hanlp", standards_pos),

for model in model_list:
    res = process_untokenized_sents(sentences, model) # list of lists of tok pos tuple
    # average_length = sum([len(tok_pos_list) for tok_pos_list in res])/50
    # print(average_length)
    # save unconverted pos results
    res_df = pd.DataFrame(res, index=range(1,51))
    res_df.to_csv("model_results/{}_pos_all.csv".format(model))

    res_poss = []
    for tok_pos_list in res:
        pos_list = []
        for tok, pos in tok_pos_list:
            pos_list.append(pos)
        res_poss.append(pos_list)
    res_poss = [convert_poslist_to_general(model, pos_list) for pos_list in res_poss]

    # save converted pos results
    res_poss_df = pd.DataFrame(res_poss, index=range(1,51))
    res_poss_df.to_csv("model_results/{}_pos_all_converted.csv".format(model))
    # print(res_poss)
    
    total_perc = 0
    for i in range(len(res)):
        # number of edits divided by the number of tokens in the sentence
        perc = min_edit_distance(res_poss[i], standards_pos[i])/len(res_poss[i]) 
        total_perc += perc
    
    print(model, total_perc/len(res))

# average percentage of number of edits over the number of tokens for each sentence
# jieba 0.589406834586017
# lac 0.34586625256584647
# hanlp 0.23821750030417146
# jiagu 0.3065770050886128
# thulac 0.302673435982941
# pkuseg 0.3109369136221829

