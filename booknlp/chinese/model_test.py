import pandas as pd
from min_edit_distance import min_edit_distance
from pos_match import parse_tok_pos
from pos_match import pos_mismatch
from pos_map import * # import dicts that convert pos tag sets
import opencc

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
