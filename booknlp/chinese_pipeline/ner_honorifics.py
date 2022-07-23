import pandas as pd

honorifics = pd.read_csv("chinese_evaluation/annotation/honorifics.csv")
honorifics_list = list(honorifics["simplified"]) + list(honorifics["traditional"])
# honorifics_list = list(honorifics["simplified"])

def ner_match_convert(ner, sent, offset):
    # sent is a list of tokens
    ner_string = ner[0]
    cat = ner[1]
    ner_start_idx = ner[2]
    ner_end_idx = ner[3]

    if len(sent) >= ner_end_idx:
        # if the ne is not the last token in the sentence
        next_token = sent[ner_end_idx]
        if next_token in honorifics_list:
            return [ner_string+next_token, cat, ner_start_idx+offset, ner_end_idx+offset+1]
    return [ner_string, cat, ner_start_idx+offset, ner_end_idx+offset]


if __name__ == "__main__":
    res = ner_match_convert(['雲普', 'PERSON', 0, 1], ['雲普', '叔', '坐', '在', '「', '曹', '氏', '家祠', '」', '的', '大', '門口', '，', '還', '穿', '著', '過', '冬天', '的', '那', '件', '破舊', '棉袍', '；', '身子', '微微', '顫動', '，', '像', '是', '耐', '不', '住', '這', '襲', '人', '的', '寒氣', '。'])
    print(res)