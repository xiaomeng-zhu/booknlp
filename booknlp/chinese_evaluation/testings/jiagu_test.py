import jiagu

#jiagu.init() # 可手动初始化，也可以动态初始化

text = "《中國革命記》第三冊（一九一一年上海自由社編印）記載﹕辛亥九月十四日杭州府為民軍佔領﹐紹興府即日宣佈光復"

words = jiagu.seg(text) # 分词
print(words, len(words))

pos = jiagu.pos(words) # 词性标注
print(pos, len(pos))

ner = jiagu.ner(words) # 命名实体识别
print(ner, len(ner))

print(list(zip(words, pos, ner)))