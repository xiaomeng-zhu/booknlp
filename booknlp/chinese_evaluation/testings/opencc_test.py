import opencc
    
# convert into simplified chinese
converter = opencc.OpenCC('t2s.json')
with open("examples/lu_xun/ah_q_chapter12.txt", "r") as f:
    text = f.read()
simplified = converter.convert(text)

with open("examples/lu_xun/ah_q_chapter12_simp.txt", "w") as writer:
    writer.write(simplified)
# print(simplified)