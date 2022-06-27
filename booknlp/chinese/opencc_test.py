import opencc
    
# convert into simplified chinese
converter = opencc.OpenCC('t2s.json')
sentence = "收糧_v/的_u/時侯_n/，_w/衙門_nt/里_f/便_d/說_v/新_a/道爺_n/的_u/法令_n/，_w/明_d/是_v/不_vd/敢_v/要_v/錢_n/，_w/這_r/一留_n/難_d/叨蹬_v/，_w/那些_r/鄉民_n/心里_n/愿意_v/花_v/几個_m/錢_n/早早_d/了事_v"
simplified = converter.convert(sentence)
print(simplified)