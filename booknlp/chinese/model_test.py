import pandas as pd
from min_edit_distance import min_edit_distance
from pos_match import parse_tok_pos
from pos_match import pos_mismatch

def convert_pos_jiagupos_pku(poss):
    # takes in a list of pos
    map_jiagupos_pku = { # a dictionary that maps jiagu pos tag to pku
        "n":"n",
        "nt":"t",
        "nd":"f",
        "nl":"s",
        "nh":"nr",
        "nhf":"nr",
        "nhs":"nr",
        "ns":"ns",
        "nn":"n",
        "ni":"nt",
        "nz":"nz",
        "v":"v",
        "vd":"vd",
        "vl":"v",
        "vu":"v",
        "a":"a",
        "f":"b", #区别词
        "mq":"m",
        "m":"m",
        "q":"q",
        "d":"d",
        "r":"r",
        "p":"p",
        "c":"c",
        "u":"u",
        "e":"e",
        "o":"o",
        "i": "i", # 习用语
        "j":"j",
        "h":"h",
        "k":"k",
        "g":"xxx", # in original pku tagset but not used by hanlp
        "x":"x",
        "w":"w",
        "ws": "nx",
        "wu":"w",
    }
    return [map_jiagupos_pku[pos] for pos in poss]

def convert_pos_pku_jiagupos(tok_poss):
    map_pku_jiagupos = { # a dictionary that pku to jiagu pos
        "Ag":"a",
        "a":"a",
        "ad":"d", # categorize 副形词 as 副词
        "an":"n", # categorize 名形词 as 名词
        "Bg":"b",
        "b":"b",
        "c":"c",
        "Dg":"d",
        "d":"d",
        "e":"e",
        "f":"nd",
        "h":"h",
        "i":"i",
        "j":"j",
        "k":"k",
        "l":"i",
        "Mg":"m",
        "m":"m",
        "Ng":"n",
        "n":"n",
        "nr":"nh",
        "ns":"ns",
        "nt":"ni",
        "nx":"ws",
        "nz":"nz",
        "o":"o",
        "p":"p",
        "q":"q",
        "Rg":"r",
        "r":"r",
        "s":"nl",
        "Tg":"nt",
        "t":"nt",
        "u":"u",
        "Vg":"v",
        "v":"v",
        "vd":"d", # categorize 副动词 as 副词
        "vn":"n", # categorize 名动词 as 名词
        "w":"w",
        "x":"x",
        "Yg":"u", # categorize 语气词 as 助词
        "y":"u", # categorize 语气词 as 助词
        "z":"a",
    }
    return [(tok, map_pku_jiagupos[pos]) for tok, pos in tok_poss]

""" input: list of sentences to tokenize
output: list of lists of token strings """
def jieba_process_all(sentences, tok=True, pos=True):
    # cannot provide pos based on tokenized result
    import jieba
    import jieba.posseg as pseg
    import paddle
    paddle.enable_static()
    jieba.enable_paddle()

    all_res = []
    for sent in sentences:
        tok_pos_list = pseg.cut(sent,use_paddle=True) # returns list of (tok, pos)
        if pos:
            all_res.append(tok_pos_list)
        else:
            sent_toks = [tok for tok, _ in tok_pos_list]
            all_res.append(sent_toks)
    
    return all_res

""" input & output same as above """
def lac_process_all(sentences, tok=True, pos=True):
    # cannot provide pos based on tokenized result
    from LAC import LAC

    lac = LAC(mode='lac')
    tok_pos_list = lac.run(sentences) # returns list of (tok_list, pos_list)

    if tok:
        all_toks = [tok_list for tok_list, _ in tok_pos_list]
        return all_toks
    else:
        return [zip(tok_list, pos_list) for tok_list, pos_list in tok_pos_list]

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
            all_toks_pos = HanLP(tokens=all_toks, tasks='pos/pku')
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
            return [jiagu.pos(toks) for toks in all_toks]
    else: # perform only pos tagging
        assert pos and tokenized != []
        all_pos = [jiagu.pos(toks) for toks in tokenized]

        all_pos_converted = all_pos # comment this out if convert to pku for comparison
        # all_pos_converted = list(map(convert_pos_jiagupos_pku, all_pos))
        
        for i in range(len(tokenized)):
            all_toks_pos.append(list(zip(tokenized[i], all_pos_converted[i])))
        # print(all_toks_pos)
        return all_toks_pos

""" input & output same as above """
def thulac_process_all(sentences, tok=True, pos=True):
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
    
    return all_res

""" input & output same as above """
def pkuseg_process_all(sentences, tok=True, pos=True):
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
    
if __name__ == '__main__':
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
    with open("annotation_50.txt", "r") as f:
        sentences = f.readlines()
    sentences = [sent.rstrip('\n') for sent in sentences]

    # gold standard tokenized result
    standards_tok = pd.read_csv("annotation/standard_tokenization.csv")
    standards_tok = list(standards_tok.iloc[:, 1]) # list of strings with tokens separated by / 
    standards_tok = [sent_string.split("/") for sent_string in standards_tok]

    # gold standard tok and pos result
    standards_tok_pos = pd.read_csv("annotation/standard_pos.csv")
    standards_tok_pos = list(standards_tok_pos.iloc[:, 1])
    standards_tok_pos = parse_tok_pos(standards_tok_pos)
    
    standards_tok_pos_converted = list(map(convert_pos_pku_jiagupos, standards_tok_pos))
    
    for model in model_list:
        res = process_tokenized_sents(sentences, model, standards_tok)
        if model == "jiagu":
            total_mismatch, average_mismatch, mismatch_list = pos_mismatch(standards_tok_pos_converted, res)
        else:
            total_mismatch, average_mismatch, mismatch_list = pos_mismatch(standards_tok_pos, res)
        
        print(model, total_mismatch, average_mismatch)
        mismatch_df = pd.DataFrame(mismatch_list, columns=["sent_idx", "tok", "standard", model]).to_csv(model+"_pos_mismatch_tokens.csv")

    # average percent mismatch when convert jiagupos to pku
    # hanlp 253 0.16073418054397515
    # jiagu 428 0.2694341499643043

    # average percent mismatch when convert pku to jiagupos
    # hanlp 253 0.16073418054397515
    # jiagu 458 0.28878558958873424


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
#     df.to_csv("model_results_50.csv")
        
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