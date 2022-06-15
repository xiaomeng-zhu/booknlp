import pandas as pd
from min_edit_distance import min_edit_distance

""" input: list of sentences to tokenize
output: list of lists of token strings """
def jieba_tok_all(sentences):
    import jieba
    import jieba.posseg as pseg
    import paddle
    paddle.enable_static()
    jieba.enable_paddle()
    all_toks = []
    for sent in sentences:
        tok_pos_list = pseg.cut(sent,use_paddle=True) # returns list of (tok, pos)
        sent_toks = [tok for tok, _ in tok_pos_list]
        all_toks.append(sent_toks)
    return all_toks

""" input & output same as above """
def lac_tok_all(sentences):
    from LAC import LAC
    lac = LAC(mode='lac')
    tok_pos_list = lac.run(sentences) # returns list of (tok_list, pos_list)
    all_toks = [tok_list for tok_list, _ in tok_pos_list]
    return all_toks

""" input & output same as above """
def hanlp_tok_all(sentences):
    from hanlp_restful import HanLPClient
    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    all_toks = HanLP.tokenize(sentences) # returns list of lists of token strings
    return all_toks

""" input & output same as above 
    NOTE: encountered tokenizers package version conflict """
def stanza_tok_all(sentences):
    import stanza
    nlp = stanza.Pipeline('zh', processor="tokenize")
    all_toks = []
    for sent in sentences:
        doc = nlp(sent)
        all_toks.append(doc.tokens)
    return all_toks

""" input & output same as above """
def jiagu_tok_all(sentences):
    import jiagu
    all_toks = [jiagu.seg(sent) for sent in sentences]
    return all_toks

""" input & output same as above """
def thulac_tok_all(sentences):
    import thulac
    import time
    if not hasattr(time, 'clock'): # resolve deprecated method use 
        setattr(time,'clock',time.perf_counter)
    thu1 = thulac.thulac(seg_only=False)
    all_toks = []
    for sent in sentences:
        tok_pos_str = thu1.cut(sent, text=True)  # returns string of "tok1_pos1 tok2_pos2" separated by whitespace
        tok_pos_list = tok_pos_str.split(" ") # list of "tok1_pos1", "tok2_pos2"
        toks = []
        for tok_pos in tok_pos_list:
            toks.append(tok_pos.split("_")[0])
        all_toks.append(toks)
    return all_toks

""" input & output same as above """
def pkuseg_tok_all(sentences):
    import pkuseg
    seg = pkuseg.pkuseg(postag=True)           # load model using default params
    all_toks = []
    for sent in sentences:
        tok_pos_list = seg.cut(sent) # returns list of (tok, pos)
        tok_list = [tok for tok, _ in tok_pos_list]
        all_toks.append(tok_list)
    return all_toks

""" input & output same as above 
    NOTE: encountered tokenizers package version conflict """
def ltp_tok_all(sentences):
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
def tokenize_all_sents(sentences, model):
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

def output_sents_txt(file_name, tokenized, model, score):
    with open(file_name, "a") as f:
        f.write(model+"\n")
        for sent in tokenized:
            f.write("/".join(sent)+"\n")
        f.write("The {} tokenizer results in {} number of edits\n".format(model, score))
        f.write("\n")

def predict(sentence, model="jieba"):
    """predict segments and part of speech by using different model.
    """
    if model == "jieba":
        import jieba
        import jieba.posseg as pseg
        import paddle
        paddle.enable_static()
        jieba.enable_paddle()
        words = pseg.cut(sentence,use_paddle=True) # paddle mode
        words_list = []
        for word, flag in words:
            words_list.append((word,flag))
        return words_list
    elif model == "lac":
        from LAC import LAC
        lac = LAC(mode='lac')
        sent_list = lac.run(sentence) # returns (word_list, tags_list)
        words_list = []
        for i in range(len(sent_list[0])):
            words_list.append((sent_list[0][i], sent_list[1][i]))
        return words_list
    elif model == "hanlp": # commented out because of restriction of request frequency
        from hanlp_restful import HanLPClient
        HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
        # if auth is None then it is connected to the server anonymously
        # language = 'zh' for chinese, language = 'mul' for multi languages
        hlp = HanLP.tokenize(sentence)
        pos_res = HanLP(tokens=hlp, tasks='pos/863') # a dictionary
        words_list = []
        for i, sent in enumerate(pos_res["tok"]):
            for j in range(len(sent)):
                words_list.append((pos_res["tok"][i][j], pos_res["pos/863"][i][j]))
        return words_list
    elif model == "stanza": # model uses CTB tagging for pos
        import stanza
        nlp = stanza.Pipeline('zh') 
    elif model == "jiayan": # expects better performance for classical texts
        from jiayan import load_lm
        from jiayan import CharHMMTokenizer
        from jiayan import CRFPOSTagger
        lm = load_lm('jiayan_model/jiayan.klm')
        tokenizer = CharHMMTokenizer(lm)
        words = list(tokenizer.tokenize(sentence))
        # return " ".join(list(tokenizer.tokenize(sentence)))
        postagger = CRFPOSTagger()
        postagger.load('jiayan_model/pos_model')
        poss = postagger.postag(words)
        words_list = []
        for i in range(len(words)):
            words_list.append((words[i], poss[i]))
        return words_list
    elif model == "jiagu":
        import jiagu
        jiagu_res = jiagu.seg(sentence)
        jiagu_pos = jiagu.pos(jiagu_res)
        words_list = [(jiagu_res[i], jiagu_pos[i]) for i in range(len(jiagu_res))]
        return words_list
    elif model== "ltp": # in ltp_test.py
        from ltp import LTP
        ltp = LTP()
        segment, hidden = ltp.seg([sentence])
        ltp_res = ""
        for sentList in segment:
            ltp_res += " ".join(sentList)
        return ltp_res
    elif model == "thulac":
        import thulac
        import time
        if not hasattr(time, 'clock'): # resolve deprecated method use 
            setattr(time,'clock',time.perf_counter)
        thu1 = thulac.thulac(seg_only=False)
        text = thu1.cut(sentence, text=True)  # tokenize the sentence
        tokens = text.split(" ")
        words_list = []
        for token in tokens:
            words_list.append((token.split("_")[0], token.split("_")[1]))
        return words_list
    elif model == "pkuseg":
        import pkuseg
        seg = pkuseg.pkuseg(postag=True)           # load model using default params
        text = seg.cut(sentence)
        return text


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
        "jieba",
        "lac",
        "hanlp",
        "stanza", # tokenizer package version conflict
        # "jiayan", # model too large to be committed
        "jiagu",
        # "ltp", # tokenizer package version conflict
        "thulac",
        "pkuseg",
    ]

    # previous example sentences
    #     sentences = [
# #         "是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。", # zhuangzi
# #         '''并題一絕云：
# # 　　滿紙荒唐言，一把辛酸淚！
# # 　　都云作者痴，誰解其中味？
# # 　　出則既明，且看石上是何故事．''',
# #         "吳媽﹐是趙太爺家裡唯一的女僕﹐洗完了碗碟﹐也就在長凳上坐下了﹐而且和阿Ｑ談閒天﹕“太太兩天沒有吃飯哩﹐因為老爺要買一個小的……”", # ah-q
#         "前幾天，狼子村的佃戶來告荒，對我大哥說，他們村裡的一個大惡人，給大家打死了；幾個人便挖出他的心肝來，用油煎炒了吃，可以壯壯膽子。", # diary of a madman
#     ]

    with open("annotation_50.txt", "r") as f:
        sentences = f.readlines()
    sentences = [sent.rstrip('\n') for sent in sentences]

    res_list = [] # list of scores for each model

    # read from gold standard result
    standards = pd.read_csv("annotation/standard_tokenization.csv")
    standards = list(standards.iloc[:, 1])

    all_model_sents = []

    for model in model_list:
        tokenized, score = calculate_score(sentences, standards, model)
        all_model_sents.append(tokenized) # nested list of model and sentence
        output_sents_txt("model_results_50.txt", tokenized, model, score)
        print(model, score)
    
    # output csv file for all tokenized results of all models
    df = pd.DataFrame(all_model_sents, index=model_list, columns=range(1,51))
    df.to_csv("model_results_50.csv")
        
    # for sentence in sentences:
    #     model_res = [] # list of results for each model
    #     for model in model_list:
    #         tokenized = tokenize(sentence, model)
    #         print(tokenized)
    #         model_res.append((model, tokenized))
    #     res_list.append(model_res)
    # print(res_list)
    
    # output text file for all models of all sentences
    # with open('model_results.txt', 'w') as writer:
    #     for i in range(len(sentences)):
    #         writer.write(sentences[i]+"\n")
    #         for model, tokens in res_list[i]:
    #             writer.write(model+"\t")
    #             sent = ""
    #             for token_pos in tokens:
    #                 sent += ("".join(token_pos)+"/ ")
    #             writer.write(sent+"\n")
    #         writer.write("\n")