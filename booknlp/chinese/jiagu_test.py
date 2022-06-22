import jiagu

#jiagu.init() # 可手动初始化，也可以动态初始化

text = "民众应该尊重现执政当局的权威。"

words = jiagu.seg(text) # 分词
print(words)

pos = jiagu.pos(words) # 词性标注
print(pos)

ner = jiagu.ner(words) # 命名实体识别
print(ner)