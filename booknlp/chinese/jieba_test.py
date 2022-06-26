import jieba
import jieba.posseg as pseg
import paddle
paddle.enable_static()
jieba.enable_paddle()

sentence = "这个软件会泄露用户隐私吗？"
all_res = []
obj = pseg.cut(sentence,use_paddle=True)
tok_pos_list = [(tok, pos) for tok, pos in obj]
print(tok_pos_list)