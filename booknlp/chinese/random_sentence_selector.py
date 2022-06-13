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

if __name__ == '__main__':
    book_name = "zhaohuaxishi"
    file_name = "examples/lu_xun/"+book_name+".txt"
    doc = read_from_txt(file_name)
    clean_doc = strip_header_footer(doc)
    sents = split_by_sentence(clean_doc)

    selected = random_select(5, sents, 30)
    print(selected)
    with open("annotation_50.txt", "a") as writer:
        writer.write("\n朝花夕拾追加：\n")
        for sent in selected:
            writer.write(sent+"\n")