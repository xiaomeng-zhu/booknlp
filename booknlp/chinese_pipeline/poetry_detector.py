import re
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

def split_by_punc(doc):
    quote_punc = ["「", "」", "“" ,"”", "\"", "『", "』", "`", "'"]
    punc = ["。", "！", "？", "，", "， ", "．", ". ", "：", " ", "\u3000"]
    sents = re.split('|'.join(quote_punc + punc), doc)

    sents = [sent for sent in sents if sent != ""]
    # sents = doc.split(/[''.join(quote_punc + punc)]/) # turns lists into string for RE
    return sents

def identify_start(prev_sent, curr_sent, next_sent):
    if (prev_sent == None or len(prev_sent) != 5) and len(curr_sent) == 5 and len(next_sent) == 5:
        return True
    elif (prev_sent == None or len(prev_sent) != 7) and len(curr_sent) == 7 and len(next_sent) == 7:
        return True
    else:
        return False

def identify_end(prev_sent, curr_sent, next_sent):
    if len(prev_sent) == 5 and len(curr_sent) == 5 and (next_sent == None or len(next_sent) != 5):
        return True
    elif len(prev_sent) == 7 and len(curr_sent) == 7 and (next_sent == None or len(next_sent) != 7):
        return True
    else:
        return False

def identify_poetry(sents):
    poems = []
    start_idx = -1
    end_idx = -1
    sents_len = len(sents)
    # for idx, sent in enumerate(sents):
    #     if idx != 0 and idx != sents_len - 1: # avoid index out of bounds error
    #         prev_sent = sents[idx-1]
    #         next_sent = sents[idx+1]
    #         if identify_start(prev_sent, sent, next_sent) and (start_idx == -1):
    #             start_idx = idx
    #         if identify_end(prev_sent, sent, next_sent) and (end_idx == -1):
    #             end_idx = idx
    #             poem_length = end_idx - start_idx + 1
    #             if poem_length >= 4:
    #                 poems.append(sents[start_idx:end_idx+1])
    #             start_idx = -1
    #             end_idx = -1
    for idx, sent in enumerate(sents):
        if idx == 0:
            prev_sent = None
            next_sent = sents[idx+1]
        elif idx == sents_len - 1:
            prev_sent = sents[idx-1]
            next_sent = None
        else:
            prev_sent = sents[idx-1]
            next_sent = sents[idx+1]
        if (next_sent != None) and identify_start(prev_sent, sent, next_sent) and (start_idx == -1):
            start_idx = idx # start found
        if (prev_sent != None) and identify_end(prev_sent, sent, next_sent) and (end_idx == -1):
            end_idx = idx # end found
            poem_length = end_idx - start_idx + 1
            if poem_length >= 4: # if long enough to count as a poem
                poems.append(sents[start_idx:end_idx+1])
            start_idx = -1
            end_idx = -1
    return poems

def extract_and_output_poetry(doc, text_title):
    splitted_list = split_by_punc(doc)
    poems = identify_poetry(splitted_list)
    poems = [poem for poem in poems if len(poem) % 2 == 0]
    with open("chinese_pipeline/outputs/{}_poetry.txt".format(text_title), "w"):
        for poem in poems:
            writer.write(" ".join(poem))
            writer.write("\n")


if __name__ == '__main__':
    book_name = "zhaohuaxishi"
    file_name = "examples/with_poetry/"+book_name+".txt"
    doc = text_file_to_string(file_name)
    clean_doc = strip_header_footer(doc)
    splitted_list = split_by_punc(clean_doc)
    poems = identify_poetry(splitted_list)
    odd = 0
    even = 0
    for poem in poems:
        if len(poem) % 2 == 0:
            even += 1
        else:
            odd += 1
            # print(poem)
    print(odd, even)
    print(len(poems))

    # output
    with open("book_output/doc_"+book_name+".txt","w") as writer:
        writer.write("\n".join(splitted_list))
    with open("book_output/poem_"+book_name+".txt", "w") as writer:
        for poem in poems:
            writer.write(" ".join(poem))
            writer.write("\n")

    