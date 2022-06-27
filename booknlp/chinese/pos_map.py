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
    "ws":"nx",
    "wu":"w",
}

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

map_jiebapos_general = {
    "a":"a",
    "ad":"d",
    "an":"n",
    "c":"c",
    "d":"d",
    "f":"f",
    "m":"m",
    "n":"n",
    "nr":"nr",
    "ns":"ns",
    "nt":"nt",
    "nw":"n",
    "nz":"nz",
    "p":"p",
    "q":"q",
    "r":"r",
    "s":"s",
    "t":"t",
    "u":"u",
    "v":"v",
    "vd":"d", # need to double check
    "vn":"n", # need to double check
    "w":"w",
    "xc":"x",
    "PER":"nr",
    "LOC":"ns",
    "ORG":"nt",
    "TIME":"t",
}

map_lacpos_general = {
    "a":"a",
    "ad":"d",
    "an":"n",
    "c":"c",
    "d":"d",
    "f":"f",
    "m":"m",
    "n":"n",
    "PER":"nr",
    "LOC":"ns",
    "ORG":"nt",
    "nw":"n",
    "nz":"nz",
    "p":"p",
    "q":"q",
    "r":"r",
    "s":"s",
    "TIME":"t",
    "u":"u",
    "v":"v",
    "vd":"d", # need to double check
    "vn":"n", # need to double check
    "w":"w",
    "xc":"x",
    "": "x", # ('先前──', '')
}

map_hanlppos_general = {
    # also used to transform gold standard
    "Ag":"a",
    "a":"a",
    "ad":"d",
    "an":"n",
    "Bg":"a",
    "b":"a",
    "c":"c",
    "Dg":"d",
    "d":"d",
    "e":"x",
    "f":"f",
    "h":"h",
    "i":"i",
    "j":"j",
    "k":"k",
    "l":"i",
    "Mg":"m",
    "m":"m",
    "Ng":"n",
    "n":"n",
    "nr":"nr",
    "ns":"ns",
    "nt":"nt",
    "nx":"n",
    "nz":"n",
    "o":"o",
    "p":"p",
    "q":"q",
    "Rg":"r",
    "r":"r",
    "s":"s",
    "Tg":"t",
    "t":"t",
    "u":"u",
    "v":"v",
    "Vg":"v",
    "vd":"d",
    "vn":"n",
    "w":"w",
    "x":"x",
    "Yg":"x",
    "y":"x",
    "z":"a",
}

map_jiagupos_general = {
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
    "vd":"d",
    "vl":"v",
    "vu":"v",
    "a":"a",
    "f":"a",
    "mq":"m",
    "m":"m",
    "q":"q",
    "d":"d",
    "r":"r",
    "p":"p",
    "c":"c",
    "u":"u",
    "e":"x",
    "o":"o",
    "i":"i",
    "j":"j",
    "h":"h",
    "k":"k",
    "g":"g",
    "x":"x",
    "w":"w",
    "ws":"n",
    "wu":"w",
}

map_thulacpos_general = {
    "a":"a",
    "c":"c",
    "d":"d",
    "e":"e",
    "f":"f",
    "g":"g",
    "h":"h",
    "i":"i",
    "j":"j",
    "k":"k",
    "m":"m",
    "mq":"m",
    "n":"n",
    "np":"nr",
    "ns":"ns",
    "ni":"nt",
    "nz":"nz",
    "o":"o",
    "p":"p",
    "q":"q",
    "r":"r",
    "s":"s",
    "t":"t",
    "u":"u",
    "v":"v",
    "w":"w",
    "x":"x",
    "y":"x",
    "id":"NA", # '開談判_id'
}

map_pkusegpos_general = {
    "Ag":"a",
    "a":"a",
    "ad":"d", # categorize 副形词 as 副词
    "an":"n", # categorize 名形词 as 名词
    "Bg":"a",
    "b":"a",
    "c":"c",
    "Dg":"d",
    "d":"d",
    "e":"x",
    "f":"f",
    "h":"h",
    "i":"i",
    "j":"j",
    "k":"k",
    "l":"i",
    "Mg":"m",
    "m":"m",
    "Ng":"n",
    "n":"n",
    "nr":"nr",
    "ns":"ns",
    "nt":"nt",
    "nx":"n",
    "nz":"nz",
    "o":"o",
    "p":"p",
    "q":"q",
    "Rg":"r",
    "r":"r",
    "s":"s",
    "Tg":"t",
    "t":"t",
    "u":"u",
    "Vg":"v",
    "v":"v",
    "vd":"d", # categorize 副动词 as 副词
    "vn":"n", # categorize 名动词 as 名词
    "w":"w",
    "x":"x",
    "Yg":"x", 
    "y":"x",
    "z":"a",
}
    
def convert_pos_jiagupos_pku(tok_poss):
    # takes a list of tok pos tuples
    return [(tok, map_jiagupos_pku[pos]) for tok, pos in tok_poss]

def convert_pos_pku_jiagupos(tok_poss):
    # takes a list of tok pos tuples
    return [(tok, map_pku_jiagupos[pos]) for tok, pos in tok_poss]

def convert_tokposlist_to_general(model, tok_poss):
    # used in pos mismatch percentage calculation
    # takes a list of tok pos tuples
    if model == "hanlp":
        return [(tok, map_hanlppos_general[pos]) for tok, pos in tok_poss]
    elif model == "jiagu":
        return [(tok, map_jiagupos_general[pos]) for tok, pos in tok_poss]

def convert_poslist_to_general(model, pos_list):
    # used in min_edit_distance calculation
    # takes a list of pos tags
    if model == "jieba":
        return [map_jiebapos_general[pos] for pos in pos_list]
    elif model == "lac":
        return [map_lacpos_general[pos] for pos in pos_list]
    elif model == "hanlp":
        return [map_hanlppos_general[pos] for pos in pos_list]
    elif model == "jiagu":
        return [map_jiagupos_general[pos] for pos in pos_list]
    elif model == "thulac":
        return [map_thulacpos_general[pos] for pos in pos_list]
    elif model == "pkuseg":
        return [map_pkusegpos_general[pos] for pos in pos_list]
    else:
        return None

if __name__ == "__main__":
    print(len(map_jiebapos_general))

