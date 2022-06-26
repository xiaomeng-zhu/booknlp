from hanlp_restful import HanLPClient
import opencc
# convert into simplified chinese
converter = opencc.OpenCC('t2s.json')

sentence = """黛玉一一的都答應著．只見一個丫鬟來回：“老太太那里傳晚飯了。”
王夫人忙攜黛玉從后房門由后廊往西，出了角門，是一條南北寬夾道．南邊
是倒座三間小小的抱廈廳，北邊立著一個粉油大影壁，后有一半大門，小小
一所房室．王夫人笑指向黛玉道：“這是你鳳姐姐的屋子，回來你好往這里
找他來，少什么東西，你只管和他說就是了。”這院門上也有四五個才總角
的小廝，都垂手侍立．王夫人遂攜黛玉穿過一個東西穿堂，便是賈母的后院
了．于是，進入后房門，已有多人在此伺候，見王夫人來了，方安設桌椅．
賈珠之妻李氏捧飯，熙鳳安箸，王夫人進羹．賈母正面榻上獨坐，兩邊四張
空椅，熙鳳忙拉了黛玉在左邊第一張椅上坐了，黛玉十分推讓．賈母笑道：
“你舅母你嫂子們不在這里吃飯．你是客，原應如此坐的。”黛玉方告了座
，坐了．賈母命王夫人坐了．迎春姊妹三個告了座方上來．迎春便坐右手第
一，	探春左第二，惜春右第二．旁邊丫鬟執著拂塵，漱盂，巾帕．李，
鳳二人立于案旁布讓．外間伺候之媳婦丫鬟雖多，卻連一聲咳嗽不聞．寂然飯
，各有丫鬟用小茶盤捧上茶來．當日林如海教女以惜福養身，云飯后務待飯
粒咽盡，過一時再吃茶，方不傷脾胃．今黛玉見了這里許多事情不合家中之
式，不得不隨的，少不得一一改過來，因而接了茶．早見人又捧過漱盂來，
黛玉也照樣漱了口．□手畢，又捧上茶來，這方是吃的茶．賈母便說：“你
們去罷，讓我們自在說話儿。”王夫人听了，忙起身，又說了兩句閒話，方
引鳳，李二人去了．賈母因問黛玉念何書．黛玉道：“只剛念了《四書》。
”黛玉又問姊妹們讀何書．賈母道：“讀的是什么書，不過是認得兩個字，
不是睜眼的瞎子罷了！”"""
sentence = converter.convert(sentence)
print(sentence)

HanLP = HanLPClient('https://www.hanlp.com/api', auth="MTE0NkBiYnMuaGFubHAuY29tOlZWSDJwMWRtdW85cjNKMTI=", language='zh') 
# if auth is None then it is connected to the server anonymously
# language = 'zh' for chinese, language = 'mul' for multi languages
# hlp = HanLP.tokenize(sentence)
# pos_res = HanLP(tokens=hlp, tasks='pos/863') # a dictionary
HanLP(sentence, tasks='ner').pretty_print()