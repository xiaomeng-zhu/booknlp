from hanlp_restful import HanLPClient
import opencc
import time
converter = opencc.OpenCC('t2s.json')

with open("examples/with_poetry/hongloumeng.txt", "r") as f:
  sentence = f.read()
sentence = converter.convert(sentence)
# print(sentence)
time0 = time.perf_counter()
HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
# if auth is None then it is connected to the server anonymously
# language = 'zh' for chinese, language = 'mul' for multi languages
# hlp = HanLP.tokenize(sentence)
# pos_res = HanLP(tokens=hlp, tasks='pos/863') # a dictionary

# print(HanLP.coreference_resolution(sentence))
all_toks = HanLP.tokenize(sentence)
time1 = time.perf_counter()
print(time1-time0)
print(all_toks[:200])


"""
[
  [
    ['老太太', 14, 15], 
    ['王夫人', 21, 23], 
    ['王夫人', 75, 77], 
    ['王夫人', 131, 133], 
    ['贾母', 142, 144], 
    ['母', 143, 144], 
    ['王夫人', 163, 165], 
    ['王夫人', 183, 185], 
    ['贾母', 188, 190], 
    ['母', 189, 190], 
    ['贾母', 219, 220], 
    ['贾母', 253, 254], 
    ['王夫人', 255, 257], 
    ['贾', 423, 424], 
    ['贾母', 423, 425], 
    ['母', 424, 425], 
    ['王夫人', 441, 443], 
    ['贾', 466, 467], 
    ['贾母', 466, 468], 
    ['母', 467, 468], 
    ['贾', 497, 498], 
    ['贾母', 497, 499], 
    ['母', 498, 499]
  ], 
  [
    ['黛玉', 0, 1], 
    ['黛玉', 25, 26], 
    ['黛玉', 79, 80], 
    ['你', 85, 86], 
    ['你', 92, 93], 
    ['你', 104, 105], 
    ['黛玉', 135, 136], 
    ['黛玉', 205, 206], 
    ['黛玉', 215, 216], 
    ['你', 224, 225], 
    ['你', 226, 227], 
    ['你', 234, 235], 
    ['黛玉方', 245, 246], 
    ['黛玉', 364, 365], 
    ['黛玉', 400, 401], 
    ['黛玉', 470, 471], 
    ['黛玉', 475, 476], 
    ['黛玉', 488, 489]
  ], 
  [
    ['这', 83, 84], 
    ['这里', 95, 96]
  ], 
  [
    ['你凤姐姐', 85, 88], 
    ['他', 97, 98], 
    ['他', 107, 108]
  ], 
  [
    ['贾母的后院', 142, 146], 
    ['母的后院', 143, 146], 
    ['此', 159, 160], 
    ['这里', 231, 232]
  ], 
  [
    ['后房门', 27, 29], 
    ['后房门', 151, 153]
  ], 
  [
    ['贾', 142, 143], 
    ['贾', 188, 189]
  ], 
  [
    ['熙凤', 179, 180], 
    ['熙凤', 201, 202]
  ], 
  [
    ['座', 248, 249], 
    ['座', 266, 267]
  ], 
  [
    ['迎春', 260, 261], 
    ['迎春', 270, 271]
  ], 
  [
    ['茶', 333, 334], 
    ['茶', 388, 389], 
    ['茶', 413, 414]
  ], 
  [
    ['你们', 429, 431], 
    ['我们', 435, 436]
  ], 
  [
    ['李，凤二人', 297, 302], 
    ['凤，李二人', 458, 463]
  ], 
  [
    ['姊妹', 491, 492], 
    ['姊妹们', 491, 493]
  ]
]"""