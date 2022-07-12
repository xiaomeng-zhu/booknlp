from poetry_detector import *
import random

def split_by_sentence(doc):
    # quote_punc = ["「", "」", "“" ,"”", "\"", "『", "』", "`", "'"]
    punc = ["。", "﹗", "！", "？", "．", ". ", "\u3000"]
    sents = re.split("|".join(punc), doc)

    sents = [sent for sent in sents if sent != ""]
    return sents

def random_select(num, sents, min_len):
    # res = []
    # for sent in sents:
    #     if len(sent) > min_len:
    #         res.append(sent)
    res = [sent for sent in sents if len(sent) >= min_len]
    sample = random.sample(res, num)
    return sample

def random_text_poetry():
    # randomly select a text from the following list
    text_list = [
    "with_poetry/hongloumeng", 
    "with_poetry/jinpingmei", 
    "with_poetry/xiyouji",
    "with_poetry/huliyuanquanzhuan", 
    "with_poetry/niehaihua"
    ]
    return random.sample(text_list, 1)

def random_text_nonpoetry():
    # randomly select a text from the following list
    text_list = [
    "lu_xun/ah_q", 
    "lu_xun/zhaohuaxishi", 
    "lu_xun/kuangrenriji_with_classical", 
    "linglijiguang", 
    "fengshou", 
    ]
    return random.sample(text_list, 1)


if __name__ == '__main__':
    book_name_list = ["lu_xun/ah_q", 
    "lu_xun/zhaohuaxishi", 
    "lu_xun/kuangrenriji_with_classical", 
    "linglijiguang", 
    "fengshou", 
    "with_poetry/hongloumeng", 
    "with_poetry/jinpingmei", 
    "with_poetry/xiyouji",
    "with_poetry/huliyuanquanzhuan", 
    "with_poetry/niehaihua"]
    
    # book_name_list = ["lu_xun/kuangrenriji_with_classical"]
    non_poetry_text = random_text_nonpoetry()
    print(non_poetry_text) # ['fengshou']

    poetry_text = random_text_poetry()
    print(poetry_text) # ['with_poetry/huliyuanquanzhuan']


    # for book_name in book_name_list:
    #     file_name = "examples/"+book_name+".txt"
    #     doc = read_from_txt(file_name)
    #     clean_doc = strip_header_footer(doc)
    #     sents = split_by_sentence(clean_doc)

    #     selected = random_select(5, sents, 30)
    #     # print(selected)
    #     # print(selected)
    #     with open("annotation_50_batch2.txt", "a") as writer:
    #         for sent in selected:
    #             writer.write(sent+"\n")