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
        return jieba_process_all(sentences, True, False)
    elif model == "lac":
        return lac_process_all(sentences, True, False)
    elif model == "thulac":
        return thulac_process_all(sentences, True, False)
    elif model == "pkuseg":
        return pkuseg_process_all(sentences, True, False)
    elif model == "hanlp":
        return hanlp_process_all(sentences, True, False, []) # NOTE: this results in score of 582, much higher than before
    elif model == "jiagu":
        return jiagu_process_all(sentences, True, False, [])

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

    tokenized = tok_all_sents(sentences, model)
    tokenized_strings = []

    for i in range(len(sentences)):
        tok_str = "/".join(tokenized[i])
        tokenized_strings.append(tok_str)
        score = min_edit_distance(tok_str, standards[i])
        total += score
    return tokenized_strings, total

ner_map = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "ORG",
    "np": "PER",
    "ni": "ORG",
    "ns": "LOC",
    "nt": "ORG",
    "nr": "PER"
}

def filter_and_convert_ner(tok_pos_list):
    ners = []
    for sent_idx, tok_poss in enumerate(tok_pos_list):
        char_idx = 0
        for tok, pos in tok_poss:
            start_idx = char_idx
            end_idx = char_idx + len(tok) - 1
            if pos in ["PER", "ORG", "LOC", "np", "ns", "ni", "nt", "nr"]:
                type = ner_map[pos]
                ners.append((sent_idx+1, start_idx, end_idx, type, tok))
            char_idx += len(tok)
    return ners

def lac_ner(sentences):
    tok_pos_list = lac_process_all(sentences, True, True, []) # list of lists of tuples
    return filter_and_convert_ner(tok_pos_list)

def thulac_ner(sentences):
    tok_pos_list = thulac_process_all(sentences, True, True, []) # list of lists of tuples
    return filter_and_convert_ner(tok_pos_list)

def jiagu_ner(sentences):
    tok_pos_list = jiagu_process_all(sentences, True, True, []) # list of lists of tuples
    return filter_and_convert_ner(tok_pos_list)

def hanlp_ner(sentences):
    # assume sentences are all simplified chinese
    
    from hanlp_restful import HanLPClient

    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    
    res_sent = []
    tok_to_char_list = [] 
    for idx, sentence in enumerate(sentences):
        count = 0
        char_idx_list = [] 
        # should have the same length as the toks list, 
        # each idx is the starting idx of the first char of the token in the sentence

        ners_dict = HanLP(sentence, tasks='ner*')
        ners_list = ners_dict["ner/pku"][0]
        # print(ners_list)
        ners_tok = ners_dict["tok/fine"][0]

        for tok in ners_tok:
            char_idx_list.append(count)
            count += len(tok)
        # print(char_idx_list)

        for ner in ners_list:
            string = ner[0]
            type = ner_map[ner[1]]
            start_idx = char_idx_list[ner[2]]
            end_idx = start_idx + len(string) - 1

            # sent idx (indexing from 0), start_idx of tok, end_idx of tok, type, string
            res_sent.append([idx+1, start_idx, end_idx, type, string])

    return [tuple(ner) for ner in res_sent]

def ner_process_all(model, sentences):
    if model == "lac":
        return lac_ner(sentences)
    elif model == "thulac":
        return thulac_ner(sentences)
    elif model == "jiagu":
        return jiagu_ner(sentences)
    elif model == "hanlp":
        return hanlp_ner(sentences)



# def process_ner(model, sentences):
#     if model == "lac":
#         return lac_ner(sentences)
#     elif model == "thulac":
#         return thulac_ner(sentences)
#     elif model == "jiagu":
#         return jiagu_ner(sentences)
#     elif model == "hanlp":
#         return hanlp_ner(sentences)