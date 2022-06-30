import jieba
import jieba.posseg as pseg
import paddle
paddle.enable_static()
jieba.enable_paddle()

# sentence = "这个软件会泄露用户隐私吗？" # ('吗', 'xc')
# sentence = "两国政府正在积极谋求合作共赢。" # ('积极', 'ad')
# sentence = "妈妈是这个办公室的副主任，也是一个女司机。" # ('金', 'n') ('女', 'a')
sentence = "《红楼梦》是一本经典名著" # ('红楼梦', 'nw')
# sentence = "雨水哗啦哗啦地落下来。"
# sentence = "我对他的了解甚深" # ('甚', 'd')

all_res = []
obj = pseg.cut(sentence,use_paddle=True)
tok_pos_list = [(tok, pos) for tok, pos in obj]
print(tok_pos_list)