import pandas as pd

def parse_tok_pos(string_list):
    """
    returns a list of list of tuples of the form (tok, pos)
    """
    res = []
    for sent in string_list:
        tokens = [(tok.split("_")[0], tok.split("_")[1]) for tok in sent.split("/")]
        res.append(tokens)
    return res

def pos_mismatch(res1, res2):
    total_mismatch_perc = 0 # average percent mismatch per sentence
    mismatch_list = []
    total_mismatch = 0
    
    assert len(res1) == len(res2) # same number of sentences
    num_sent = len(res1)

    for i in range(len(res1)): # iterate through all sentences
        mismatch = 0
        sent1 = res1[i]
        sent2 = res2[i]

        if len(sent1) != len(sent2):
            print(sent1, sent2)
        assert len(sent1)==len(sent2)
        num_toks = len(sent1)

        for j in range(len(sent1)): # iterate through all tokens in the sentence
            tok1, pos1 = sent1[j]
            tok2, pos2 = sent2[j]
            assert tok1 == tok2 # the two tokens for comparison must be the same one
            if pos1 != pos2:
                mismatch_list.append((i+1, tok1, pos1, pos2))
                mismatch += 1

        perc = mismatch/num_toks # percent mismatch of the current sentence = number of mismatch / number of tokens
        total_mismatch += mismatch
        total_mismatch_perc += perc
    
    average_mismatch = total_mismatch_perc/num_sent

        
    
    return total_mismatch, average_mismatch, mismatch_list

if __name__ == "__main__":
    m_pos = pd.read_csv("annotation/miranda_pos.csv")
    k_pos = pd.read_csv("annotation/kiara_pos.csv")

    m_pos = list(m_pos.iloc[:, 1])
    k_pos = list(k_pos.iloc[:, 1])
    # print(m_pos[:10])

    m_tok_pos = parse_tok_pos(m_pos)
    k_tok_pos = parse_tok_pos(k_pos)

    agreement, average_mismatch, mistmatch_list = pos_mismatch(m_tok_pos, k_tok_pos)
    print("There are {} total mismatches of POS tagging".format(agreement))
    print("The average percentage of mismatch per sentence is {}.".format(average_mismatch))
    mismatch_df = pd.DataFrame(mistmatch_list, columns=["sent_idx", "tok", "m_tag", "k_tag"]).to_csv("pos_mismatch_tokens.csv")

    # There are 233 total mismatches of POS tagging
    # The average percentage of mismatch per sentence is 0.14554441853443462.