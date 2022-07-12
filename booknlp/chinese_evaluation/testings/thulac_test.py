import thulac	
import time
if not hasattr(time, 'clock'): # resolve deprecated method use 
    setattr(time,'clock',time.perf_counter)

thu1 = thulac.thulac()  #默认模式
text = thu1.cut("收糧的時侯，衙門里便說新道爺的法令，明是不敢要錢，這一留難叨蹬，那些鄉民心里愿意花几個錢早早了事", text=True)  #进行一句话分词
print(text)