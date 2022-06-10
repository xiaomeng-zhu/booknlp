def predict(sentence, model="jieba"):
    """predict segments by using different model.
    model list:
        "jieba",
        "lac",
        "thulac",
        "jiagu",
        #"pkuseg",
        "hanlp",
        "jiayan",
    Args:
        sentence: the input.
    Returns:
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
    # elif model== "ltp": # in ltp_test.py
    #     from ltp import LTP
    #     ltp = LTP()
    #     segment, hidden = ltp.seg([sentence])
    #     ltp_res = ""
    #     for sentList in segment:
    #         ltp_res += " ".join(sentList)
    #     return ltp_res
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

if __name__ == '__main__':
    import opencc
    converter = opencc.OpenCC('t2s.json')
    
    sentences_original = [
        "是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。", # zhuangzi
        '''并題一絕云：
　　滿紙荒唐言，一把辛酸淚！
　　都云作者痴，誰解其中味？
　　出則既明，且看石上是何故事．''',
        "吳媽﹐是趙太爺家裡唯一的女僕﹐洗完了碗碟﹐也就在長凳上坐下了﹐而且和阿Ｑ談閒天﹕“太太兩天沒有吃飯哩﹐因為老爺要買一個小的……”", # ah-q
        "前幾天，狼子村的佃戶來告荒，對我大哥說，他們村裡的一個大惡人，給大家打死了；幾個人便挖出他的心肝來，用油煎炒了吃，可以壯壯膽子。", # diary of a madman
    ]
    sentences = [converter.convert(sentence) for sentence in sentences_original]
    
    model_list = [
        "jieba",
        "lac", # baidu
        "hanlp",
        #"stanza",
        "jiayan",
        "jiagu",
        #"ltp",
        "thulac",
        "pkuseg",
    ]
    res_list = [] # list of results for each sentence
    for sentence in sentences:
        model_res = [] # list of results for each model
        for model in model_list:
            model_res.append((model, predict(sentence, model)))
        res_list.append(model_res)
    # print(res_list)
    
    # output text file for all models of all sentences
    with open('model_results.txt', 'w') as writer:
        for i in range(len(sentences)):
            writer.write(sentences[i]+"\n")
            for model, tokens in res_list[i]:
                writer.write(model+"\t")
                sent = ""
                for token_pos in tokens:
                    sent += ("".join(token_pos)+"/ ")
                writer.write(sent+"\n")
            writer.write("\n")

