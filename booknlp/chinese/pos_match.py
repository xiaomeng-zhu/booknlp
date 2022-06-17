import pandas as pd

def parse_tok_pos(string_list):
    """
    returns a list of list of tuples of the form (tok, pos)
    """
    res = []
    for sent in string_list:
        tokens = [(tok.split("_")[0], tok.split("_"[1])) for tok in sent.split("/")]
        res.append(tokens)
    return res

def pos_mismatch(res1, res2):
    assert len(res1) == len(res2) # same number of sentences
    total_mismatch = 0
    for i in range(len(res1)): # iterate through all sentences
        mismatch = 0
        sent1 = res1[i]
        sent2 = res2[i]
        assert len(sent1) == len(sent2) # same number of tokens for each sentence
        for j in range(len(sent1)): # iterate through all tokens in the sentence
            tok1, pos1 = sent1[j]
            tok2, pos2 = sent2[j]
            assert tok1 == tok2 # the two tokens for comparison must be the same one
            if pos1 != pos2:
                mismatch += 1
        total_mismatch += mismatch
    
    return total_mismatch

if __name__ == "__main__":
    m_pos = pd.read_csv("annotation/miranda_pos.csv")
    k_pos = pd.read_csv("annotation/kiara_pos.csv")

    m_pos = list(m_pos.iloc[:, 1])
    k_pos = list(k_pos.iloc[:, 1])

    m_tok_pos = parse_tok_pos(m_pos)
    k_tok_pos = parse_tok_pos(k_pos)

    agreement = pos_mismatch(m_tok_pos, k_tok_pos)
    print(agreement)