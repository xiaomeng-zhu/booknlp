def predict(sentence, model="jieba"):
    """predict segments by using different model.
    model list:
        jieba,
        lac,
        thulac,
        pkuseg,
        ICTCLAS,
        PyLTP,
        FNLP,
        HanLP,
        CoreNLP
    Args:
        sentence: the input.
    Returns:
    """
    if model == "jieba":
        import jieba
        return  " ".join(jieba.cut(sentence))
    elif model == "lac":
        from LAC import LAC
        lac = LAC(mode='seg')
        return " ".join(lac.run(sentence))
    elif model =="hanlp":
        from hanlp_restful import HanLPClient
        HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh') # auth不填则匿名，zh中文，mul多语种
        hlp = HanLP.tokenize(sentence)
        hlp_res = ""
        for sentList in hlp:
            hlp_res += " ".join(sentList)
        return hlp_res
    elif model == "jiagu":
        import jiagu
        jiagu_res = jiagu.seg(sentence)
        return " ".join(jiagu_res)
    elif model=="ltp":
        from ltp import LTP
        ltp = LTP()
        segment, _ = ltp.seg(["他叫汤姆去拿外衣。"])
        ltp_res = ""
        for sentList in segment:
            ltp_res += " ".join(sentList)
        return ltp_res
    elif model == "thulac":
        import thulac
        import time
        if not hasattr(time, 'clock'): # resolve deprecated method use 
            setattr(time,'clock',time.perf_counter)
        thu1 = thulac.thulac(seg_only=True)
        text = thu1.cut(sentence, text=True)  #进行一句话分词
        return text
    elif model == "pkuseg":
        import pkuseg
        seg = pkuseg.pkuseg()           # 以默认配置加载模型
        text = seg.cut(sentence)
        return " ".join(text)

if __name__ == '__main__':
    sentence = "到了姓趙的爺爺手裏，居然請了先生，教他兒子攻書，到他孫子，忽然得中一名黌門秀士。"
    model_list = [
        "jieba",
        "lac", # baidu
        "thulac",
        "jiagu",
        #"pkuseg",
        "hanlp",
    ]
    for model in model_list:
        print(model, predict(sentence, model))