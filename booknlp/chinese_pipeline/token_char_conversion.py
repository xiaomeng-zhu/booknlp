import pandas as pd

def generate_mapping_from_tokens(all_tokens):
    # a list of tokens, return a list of (character, token_id, char_id)
    res_list = []
    token_id = 0
    char_id = 0

    for token_id, token in enumerate(all_tokens):
        for char in token:
            res_list.append((char, token_id, char_id))
            char_id +=1

    return res_list

def get_token_idx_from_char(mapping_list, given_char_idx):
    # given the character index, return the token index that the character is in
    # token_idx is the second element in the 3-place tuple
    return mapping_list[given_char_idx][1]

def get_char_idx_from_token(mapping_list, given_token_idx):
    # given the token index, return the (start, end) character index pair of the token (end inclusive)
    chars = [char_idx for (char, token_idx, char_idx) in mapping_list if token_idx == given_token_idx]
    return (min(chars), max(chars))
    

if __name__ == "__main__":
    tokens = pd.read_csv("chinese_pipeline/outputs/fengshou_tokens.csv")
    tokens = tokens["token"]
    mapping = generate_mapping_from_tokens(tokens)
    # print(mapping[500:700])
    print(get_char_idx_from_token(mapping, 509))