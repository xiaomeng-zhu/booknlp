import opencc
    
# convert into simplified chinese
converter = opencc.OpenCC('t2s.json')
with open("examples/with_poetry/jinpingmei_chapter1.txt", "r") as f:
    text = f.read()
simplified = converter.convert(text)

with open("examples/with_poetry/jinpingmei_chapter1_simp.txt", "w") as writer:
    writer.write(simplified)
# print(simplified)