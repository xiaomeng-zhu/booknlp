from model_test import *

model_list = [
    "jieba",
    "lac",
    "hanlp",
    "stanza", # tokenizer package version conflict
    # "jiayan", # model too large to be committed
    "jiagu",
    # "ltp", # tokenizer package version conflict
    "thulac",
    "pkuseg",
]

with open("annotation/annotation_50.txt", "r") as f:
    sentences = f.readlines()
sentences = [sent.rstrip('\n') for sent in sentences]

res_list = [] # list of scores for each model

# read from gold standard result
standards = pd.read_csv("annotation/standard_tokenization.csv")
standards = list(standards.iloc[:, 1])

all_model_sents = []

for model in model_list:
    tokenized, score = calculate_score(sentences, standards, model)
    all_model_sents.append(tokenized) # nested list of model and sentence
    output_sents_txt("model_results_50.txt", tokenized, model, score)
    print(model, score)

# output csv file for all tokenized results of all models
df = pd.DataFrame(all_model_sents, index=model_list, columns=range(1,51))
df.to_csv("model_results/model_results_50.csv")
    
# for sentence in sentences:
#     model_res = [] # list of results for each model
#     for model in model_list:
#         tokenized = tokenize(sentence, model)
#         print(tokenized)
#         model_res.append((model, tokenized))
#     res_list.append(model_res)
# print(res_list)

# output text file for all models of all sentences
# with open('model_results.txt', 'w') as writer:
#     for i in range(len(sentences)):
#         writer.write(sentences[i]+"\n")
#         for model, tokens in res_list[i]:
#             writer.write(model+"\t")
#             sent = ""
#             for token_pos in tokens:
#                 sent += ("".join(token_pos)+"/ ")
#             writer.write(sent+"\n")
#         writer.write("\n")