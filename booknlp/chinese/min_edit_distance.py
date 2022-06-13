import pandas as pd

def clean_double_list(input_list):
    sents_clean = []
    # remove empty strings
    for sent in input_list:
        inner = []
        for token in sent:
            if token != "":
                inner.append(token)
        sents_clean.append(inner)
    return sents_clean

def min_edit_distance(sent1, sent2):
    n = len(sent1)
    m = len(sent2)

    matrix = [[i+j for j in range(m+1)] for i in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            if sent1[i-1] == sent2[j-1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    distance_score = matrix[n][m]
   
    return distance_score


if __name__ == "__main__":
    miranda = pd.read_csv("annotation/miranda_tokenization.csv")
    kiara = pd.read_csv("annotation/kiara_tokenization.csv")

    m_sents = list(miranda.iloc[:, 1])
    k_sents = list(kiara.iloc[:, 1])

    # method 1: # consider every token as a character, essentially a list of characters
    # m_sents = [sent.split("/") for sent in m_sents]
    # k_sents = [sent.split("/") for sent in k_sents]


    # remove empty strings in the list that are caused by having slashes at the end of the sentence
    # m_sents = clean_double_list(m_sents)
    # k_sents = clean_double_list(k_sents)

    # method 2: feed into a string, essentially removing and adding slashes
    # no need to split

    assert len(m_sents)==len(k_sents)

    total_score = 0
    for idx in range(len(m_sents)):
        current_score = min_edit_distance(m_sents[idx], k_sents[idx])
        print(m_sents[idx])
        print(k_sents[idx])
        print()
        total_score += current_score
    
    print("Total score is: {}".format(total_score))
    print("Average score per sentence is: {}".format(total_score/len(m_sents)))

    # method 1:
    # Total score is: 327
    # Average score per sentence is: 6.54

    # method 2:
    # Total score is: 175
    # Average score per sentence is: 3.5

