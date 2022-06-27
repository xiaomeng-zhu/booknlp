import pandas as pd
from min_edit_distance import min_edit_distance
from pos_match import parse_tok_pos
from pos_match import pos_mismatch
from pos_map import * # import dicts that convert pos tag sets

""" input: list of sentences to tokenize
output: list of lists of token strings """
def jieba_process_all(sentences, tok=True, pos=True, tokenized=[]):
    # cannot provide pos based on tokenized result
    import jieba
    import jieba.posseg as pseg
    import paddle
    paddle.enable_static()
    jieba.enable_paddle()

    all_res = []
    for sent in sentences:
        tok_pos_obj = pseg.cut(sent,use_paddle=True) # returns list of (tok, pos)
        tok_pos_list = [(tok, pos) for tok, pos in tok_pos_obj]
        if pos:
            all_res.append(tok_pos_list)
        else:
            sent_toks = [tok for tok, _ in tok_pos_list]
            all_res.append(sent_toks)
    
    return all_res

""" input & output same as above """
def lac_process_all(sentences, tok=True, pos=True, tokenized=[]):
    # cannot provide pos based on tokenized result
    from LAC import LAC

    lac = LAC(mode='lac')
    tok_pos_list = lac.run(sentences) # returns list of (tok_list, pos_list)

    if tok:
        all_toks = [tok_list for tok_list, _ in tok_pos_list]
        if not pos: # only perform tokenization
            return all_toks
        else:
            return [list(zip(tok_list, pos_list)) for tok_list, pos_list in tok_pos_list]
    
    # cannot perform only pos tagging

""" input & output same as above """
def hanlp_process_all(sentences, tok=True, pos=True, tokenized=[]):
    from hanlp_restful import HanLPClient

    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    
    all_toks = []
    all_toks_pos = []
    if tok:
        all_toks = HanLP.tokenize(sentences)
        if not pos:
            return all_toks
        else: # run pos tagging on tokens from hanlp
            all_toks_pos_dict = HanLP(tokens=all_toks, tasks='pos/pku')
            all_toks_pos = []
            # format output
            for i in range(len(sentences)):
                all_toks_pos.append(list(zip(all_toks_pos_dict["tok"][i], all_toks_pos_dict["pos/pku"][i])))
            return all_toks_pos
    else: # perform only pos tagging
        assert pos and tokenized != [] # user needs to supply outside tokenized result
        all_toks_pos_dict = HanLP(tokens=tokenized, tasks='pos/pku')
        all_toks_pos = []
        # format output
        for i in range(len(tokenized)):
            all_toks_pos.append(list(zip(all_toks_pos_dict["tok"][i], all_toks_pos_dict["pos/pku"][i])))
        # print(all_toks_pos)
        return all_toks_pos

""" input & output same as above 
    NOTE: encountered tokenizers package version conflict """
def stanza_process_all(sentences):
    import stanza
    nlp = stanza.Pipeline('zh', processor="tokenize")
    all_toks = []
    for sent in sentences:
        doc = nlp(sent)
        all_toks.append(doc.tokens)
    return all_toks

""" input & output same as above """
def jiagu_process_all(sentences, tok=True, pos=True, tokenized=[]):
    import jiagu
    all_toks = []
    all_toks_pos = []
    if tok:
        all_toks = [jiagu.seg(sent) for sent in sentences]
        if not pos:
            return all_toks
        else:
            all_pos = [jiagu.pos(toks) for toks in all_toks]
            for i in range(len(all_toks)):
                all_toks_pos.append(list(zip(all_toks[i], all_pos[i])))
            return all_toks_pos

    else: # perform only pos tagging
        assert pos and tokenized != []
        all_pos = [jiagu.pos(toks) for toks in tokenized]

        # if convert_from == "jiagu": # to pku
        #     all_pos_converted = list(map(convert_pos_jiagupos_pku, all_pos))
        # else:
        #     all_pos_converted = all_pos # comment this out if convert to pku for comparison
        
        for i in range(len(tokenized)):
            all_toks_pos.append(list(zip(tokenized[i], all_pos[i])))
        return all_toks_pos

""" input & output same as above """
def thulac_process_all(sentences, tok=True, pos=True, tokenized=[]):
    # cannot provide pos based on tokenized result. i.e. tok must be True when pos is True

    import thulac
    import time
    if not hasattr(time, 'clock'): # resolve deprecated method use 
        setattr(time,'clock',time.perf_counter)
    
    thu1 = thulac.thulac(seg_only=False)

    all_res = []
    for sent in sentences:
        tok_pos_str = thu1.cut(sent, text=True)  # returns string of "tok1_pos1 tok2_pos2" separated by whitespace
        tok_pos_list = tok_pos_str.split(" ") # list of "tok1_pos1", "tok2_pos2"
        res = []
        for tok_pos in tok_pos_list:
            if pos:
                res.append((tok_pos.split("_")[0], tok_pos.split("_")[1]))
            else:
                res.append(tok_pos.split("_")[0])
        all_res.append(res)
    
    # print(all_res)
    return all_res

""" input & output same as above """
def pkuseg_process_all(sentences, tok=True, pos=True, tokenized=[]):
    import pkuseg
    seg = pkuseg.pkuseg(postag=True)           # load model using default params

    all_res = []
    for sent in sentences:
        tok_pos_list = seg.cut(sent) # returns list of (tok, pos)
        if pos:
            all_res.append(tok_pos_list)
        else:
            tok_list = [tok for tok, _ in tok_pos_list]
            all_res.append(tok_list)
    
    return all_res

""" input & output same as above 
    NOTE: encountered tokenizers package version conflict """
def ltp_process_all(sentences):
    from ltp import LTP
    ltp = LTP()
    segment, _ = ltp.seg(sentences) # returns list of lists of (tok,pos)
    # print(ltp.seg(sentences))
    ltp_res = ""
    for sentList in segment:
        ltp_res += " ".join(sentList)
    return segment


""" inputs: list of all sentence strings to tokenize; which tok model to use 
    output: list of lists of token strings, depending on model"""
def tok_all_sents(sentences, model):
    if model == "jieba":
        return jieba_tok_all(sentences)
    elif model == "lac":
        return lac_tok_all(sentences)
    elif model == "hanlp":
        return hanlp_tok_all(sentences) # NOTE: this results in score of 582, much higher than before
    elif model == "stanza":
        return stanza_tok_all(sentences)
    elif model == "jiagu":
        return jiagu_tok_all(sentences)
    elif model == "thulac":
        return thulac_tok_all(sentences)
    elif model == "pkuseg":
        return pkuseg_tok_all(sentences)
    elif model == "ltp":
        return ltp_tok_all(sentences)
    else:
        return None

def process_untokenized_sents(sentences, model):
    if model == "jieba":
        return jieba_process_all(sentences, True, True)
    elif model == "lac":
        return lac_process_all(sentences, True, True)
    elif model == "thulac":
        return thulac_process_all(sentences, True, True)
    elif model == "pkuseg":
        return pkuseg_process_all(sentences, True, True)
    elif model == "hanlp":
        return hanlp_process_all(sentences, True, True, [])
    elif model == "jiagu":
        return jiagu_process_all(sentences, True, True, [])
    # stanza and ltp not included bc they never work

def process_tokenized_sents(sentences, model, tokenized):
    if model == "hanlp":
        return hanlp_process_all(sentences, False, True, tokenized) 
    elif model == "jiagu":
        return jiagu_process_all(sentences, False, True, tokenized)
    else:
        return None

def output_sents_txt(file_name, tokenized, model, score):
    with open(file_name, "a") as f:
        f.write(model+"\n")
        for sent in tokenized:
            f.write("/".join(sent)+"\n")
        f.write("The {} tokenizer results in {} number of edits\n".format(model, score))
        f.write("\n")

def calculate_score(sentences, standards, model):
    # calculate the average minimum edit distance for all sentences for tokenization comparison

    assert len(sentences) == len(standards)
    import opencc
    
    # convert into simplified chinese
    converter = opencc.OpenCC('t2s.json')
    sentences = [converter.convert(sentence) for sentence in sentences]

    total = 0

    tokenized = tokenize_all_sents(sentences, model)
    tokenized_strings = []

    for i in range(len(sentences)):
        tok_str = "/".join(tokenized[i])
        tokenized_strings.append(tok_str)
        score = min_edit_distance(tok_str, standards[i])
        total += score
    return tokenized_strings, total
    
# ================ for models that do not support customized tokens ================
if __name__ == '__main__':
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

    # read original sentences as untokenized strings
    with open("annotation_50.txt", "r") as f:
        sentences = f.readlines()
    sentences = [sent.rstrip('\n') for sent in sentences]

    # read gold standard tokenized result
    standards_tok = pd.read_csv("annotation/standard_tokenization.csv")
    standards_tok = list(standards_tok.iloc[:, 1]) # list of strings with tokens separated by / 
    standards_tok = [sent_string.split("/") for sent_string in standards_tok]

    # read gold standard tok and pos result
    standards_tok_pos = pd.read_csv("annotation/standard_pos.csv")
    standards_tok_pos = list(standards_tok_pos.iloc[:, 1]) # list of strings

    # get all gold standard part of speech tags
    standards_pos = [] # list of lists of pos
    for sent in standards_tok_pos:
        pos_list = []
        tok_pos_list = sent.split("/")
        for tok_pos in tok_pos_list:
            pos_list.append(tok_pos.split("_")[1]) # list of lists of pos
        standards_pos.append(pos_list)
    standards_pos_converted = map(convert_to_general, "hanlp", standards_pos),

    for model in model_list:
        res = process_untokenized_sents(sentences, model) # list of lists of tok pos tuple
        res_df = pd.DataFrame(res, index=range(1,51))
        res_df.to_csv("model_results/{}_pos_all.csv".format(model))

        res_poss = []
        for tok_pos_list in res:
            pos_list = []
            for tok, pos in tok_pos_list:
                pos_list.append(pos)
            res_poss.append(pos_list)
        res_poss = [convert_to_general(model, pos_list) for pos_list in res_poss]

        res_poss_df = pd.DataFrame(res_poss, index=range(1,51))
        res_poss_df.to_csv("model_results/{}_pos_all_converted.csv".format(model))
        # print(res_poss)
        
        total_score = 0
        for i in range(len(res)):
            score = min_edit_distance(res_poss[i], standards_pos[i])
            total_score += score
        print(model, total_score/len(res))

    # average number of edits per sentence, all converted to general for comparison
    # jieba 17.14
    # lac 9.06
    # hanlp 7.4
    # jiagu 14.1
    # thulac 11.6
    # pkuseg 11.36

# ================ for models that support customized tokens ================
# if __name__ == '__main__':
#     model_list = [
#         # "jieba",
#         # "lac",
#         "hanlp",
#         # "stanza", # tokenizer package version conflict
#         # "jiayan", # model too large to be committed
#         "jiagu",
#         # "ltp", # tokenizer package version conflict
#         # "thulac",
#         # "pkuseg",
#     ]

#     # original sentences as untokenized strings
#     with open("annotation_50.txt", "r") as f:
#         sentences = f.readlines()
#     sentences = [sent.rstrip('\n') for sent in sentences]

#     # gold standard tokenized result
#     standards_tok = pd.read_csv("annotation/standard_tokenization.csv")
#     standards_tok = list(standards_tok.iloc[:, 1]) # list of strings with tokens separated by / 
#     standards_tok = [sent_string.split("/") for sent_string in standards_tok]

#     # gold standard tok and pos result
#     standards_tok_pos = pd.read_csv("annotation/standard_pos.csv")
#     standards_tok_pos = list(standards_tok_pos.iloc[:, 1])
#     standards_tok_pos = parse_tok_pos(standards_tok_pos)

#     for model in model_list:
#         res = process_tokenized_sents(sentences, model, standards_tok)
#         direction = ""
    
#         if model == "jiagu":
#             if convert_from == "jiagu": # convert jiagu pos to pku
#                 res = list(map(convert_pos_jiagupos_pku, res))
#                 direction = "_model2pku"
#             else: # convert gold standard to jiagu pos
#                 standards_tok_pos = list(map(convert_pos_pku_jiagupos, standards_tok_pos))
#                 direction = "_standard2jiagu"

#         total_mismatch, average_mismatch, mismatch_list = pos_mismatch(standards_tok_pos, res)
        
#         print(model, total_mismatch, average_mismatch)
#         file_name = "{}_pos_mismatch{}.csv".format(model, direction)
        
        
#         mismatch_df = pd.DataFrame(mismatch_list, 
#                 columns=["sent_idx", "tok", "standard", model]).to_csv(file_name)
        

#     # average percent mismatch when convert jiagupos to pku
#     # hanlp 253 0.16073418054397515
#     # jiagu 428 0.2694341499643043

#     # average percent mismatch when convert gold standard pos in pku to jiagupos
#     # hanlp 253 0.16073418054397515
#     # jiagu 458 0.28878558958873424


# if __name__ == '__main__':
#     model_list = [
#         "jieba",
#         "lac",
#         "hanlp",
#         "stanza", # tokenizer package version conflict
#         # "jiayan", # model too large to be committed
#         "jiagu",
#         # "ltp", # tokenizer package version conflict
#         "thulac",
#         "pkuseg",
#     ]

#     with open("annotation_50.txt", "r") as f:
#         sentences = f.readlines()
#     sentences = [sent.rstrip('\n') for sent in sentences]

#     res_list = [] # list of scores for each model

#     # read from gold standard result
#     standards = pd.read_csv("annotation/standard_tokenization.csv")
#     standards = list(standards.iloc[:, 1])

#     all_model_sents = []

#     for model in model_list:
#         tokenized, score = calculate_score(sentences, standards, model)
#         all_model_sents.append(tokenized) # nested list of model and sentence
#         output_sents_txt("model_results_50.txt", tokenized, model, score)
#         print(model, score)
    
#     # output csv file for all tokenized results of all models
#     df = pd.DataFrame(all_model_sents, index=model_list, columns=range(1,51))
#     df.to_csv("model_results/model_results_50.csv")
        
#     # for sentence in sentences:
#     #     model_res = [] # list of results for each model
#     #     for model in model_list:
#     #         tokenized = tokenize(sentence, model)
#     #         print(tokenized)
#     #         model_res.append((model, tokenized))
#     #     res_list.append(model_res)
#     # print(res_list)
    
#     # output text file for all models of all sentences
#     # with open('model_results.txt', 'w') as writer:
#     #     for i in range(len(sentences)):
#     #         writer.write(sentences[i]+"\n")
#     #         for model, tokens in res_list[i]:
#     #             writer.write(model+"\t")
#     #             sent = ""
#     #             for token_pos in tokens:
#     #                 sent += ("".join(token_pos)+"/ ")
#     #             writer.write(sent+"\n")
#     #         writer.write("\n")