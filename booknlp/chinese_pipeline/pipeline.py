import re
import time
from hanlp_restful import HanLPClient
import poetry_detector

def client_set_up():
    HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
    return HanLP

def text_file_to_string(file_path):
    with open(file_path, "r") as f:
        text_string = f.read()
    return text_string

def strip_header_footer(doc):
    header_idx = -1
    footer_idx = len(doc)
    try:
        header_idx = re.search("START OF (THE|THIS) PROJECT GUTENBERG EBOOK", doc).end() # index of the last character in the header identification string
    except AttributeError:
        print("No header found.")

    # res = doc
    # if header_idx != -1: # if there is a header
    res = doc[(header_idx+1):] # if header_idx == -1, there is no header, so res is doc; otherwise, res is the slice without header

    try:
        footer_idx = re.search("End of the Project Gutenberg EBook of", res).start() # index of the first character in the footer identification string
    except AttributeError:
        print("No footer found.")
    
    res = res[:footer_idx]
    res = re.sub("\n", "", res)
    res = re.sub(re.escape("*"), "", res) # for some reason, this doesn't work when escaped * is added to punctuation list
    return res

def split_text(text_string, max_length):
    # split by sentences and into sections around max_length (exceeds max_length to split at sentence boundary)
    punc = ["。", "﹗", "！", "？", "．", ". ", "\u3000", "! ", "? ", "……"]
    #quote_punc = ["」", "”", "』", "`", "'"]
    quote_punc = ["」","”", "'", "』"]
    sections = []
    finished = False
    
    section_start = 0
    section_end = max_length

    while not finished:
        if len(text_string[section_start:]) <= (max_length / 2):
            # if what remains is less than half of max_length, append the string to the last section
            sections[-1] += text_string[section_start:]
            finished = True
        elif (len(text_string[section_start:]) > (max_length / 2)) and (len(text_string[section_start:]) <= max_length):
            # if what remains is long enough to count as a section, append to sections list
            sections.append(text_string[section_start:])
            finished = True
        else:
            # if text_string[section_end] in punc:
            #     if text_string[section_end + 1] in quote_punc:
            if text_string[section_end-1] in punc:
                if text_string[section_end] in quote_punc: # extra check for quotation mark
                    section_end += 1
                sections.append(text_string[section_start:section_end])
                # sections.append(text_string[section_start:section_end + 1])
                section_start = section_end
                # section_start = section_end + 1
                section_end = section_start + max_length
            else:
                section_end = section_end + 1

    return sections

def preprocess(text_file, text_title):
    HanLP = client_set_up()

    text_string = text_file_to_string(text_file)
    clean_string = strip_header_footer(text_string)

    poetry_detector.extract_and_output_poetry(clean_string, text_title)

    sections = split_text(clean_string, 14500) # maximum character for HanLP is 15000

    return sections

def tok(client, section):
    all_toks = client.tokenize(section)
    return all_toks

def pos():
    pass

def ner():
    pass

def honorifics():
    pass

def process(text_file, text_title):
    sections = preprocess(text_file, text_title)

    HanLP = client_set_up()
    time0 = time.perf_counter()

    tokens = []
    for section in sections:
        section_tok = tok(HanLP, section)
        tokens.append(section_tok)

    time1 = time.perf_counter()
    print(time1-time0)
    
    return tokens


if __name__ == "__main__":
    # sections = preprocess("data/fengshou_excerpt.txt")
    file_path = "chinese_evaluation/examples/fengshou.txt"
    text_title = "fengshou"
    res = process(file_path, text_title)

