def predict(sentence, model="jieba"):
    """predict segments using only the ltp model
    model list:
        "ltp"
    Args:
        sentence: the input.
    Returns:
    """
    if model== "ltp":
        from ltp import LTP
        ltp = LTP()
        segment, hidden = ltp.seg([sentence])
        # ltp_res = ""
        # for sentList in segment:
        #     ltp_res += " ".join(sentList)
        return ltp.pos(hidden)
    

if __name__ == '__main__':
    sentences = ["是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。"]
    model_list = [
        "ltp"
    ]
    for sentence in sentences:
        for model in model_list:
            print(model, predict(sentence, model))
    
    # output text file for all models of all sentences
    with open('model_results.txt', 'w') as writer:
        # flesh this out
        writer.write("")
