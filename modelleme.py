import nltk
import re
from gensim.models import Word2Vec

text = "Alışveriş merkezlerinin açılması, sosyal mesafe ile maske kullanımı için büyük bir önem taşıyor. Sağlık Bakanı Dr. Fahrettin KOCA, insan yoğunluğunun bir anda artmasına ilişkin bazı konuların altını çizdi. AVM’lerin açılması ile birlikte yaşanan izdiham, Sağlık Bakanlığı tarafından dikkatli bir şekilde takip edildi. Özellikle ısıtma ve soğutma sistemlerinin hatasız bir şekilde çalışmasını isteyen Sağlık Bakanlığı, bu konudaki çalışmalarını sürdürüyor. Maske kullanımı da bu kapsam altında inceleniyor. Dr. Fahrettin KOCA, paylaştığı mesajda şu sözlere yer verdi: “Yakın mesafedeki iki kişiden ikisi de maske takmış olsa, virüs maskeye rağmen birinden diğerine bulaşabilir. Ya hareketli, kalabalık yerlerde? Evden çıkarken mutlaka maske takın. Ama maske SOSYAL MESAFE olmadan Koronavirüs riskinden sizi uzak tutmaya yetmeyebilir. Risk almayın.”"

# Preprocessing
text = re.sub(r"\[\d+\]"," ",text)
text = re.sub(r"\["," ",text)
text = re.sub(r"\]"," ",text)
text = re.sub(r"\("," ",text)
text = re.sub(r"\)"," ",text)
text = re.sub(r"[:,'\"-]"," ",text)
text = re.sub(r"\s+"," ",text)
text = text.strip()

sentences = nltk.sent_tokenize(text)
sents = []

for a in sentences:
    sents.append(nltk.word_tokenize(a))

model = Word2Vec(sents,min_count=1)
vector = model.wv['maske','virüs','sosyal','mesafe','önem']
print(vector)

model.save('model.bin')
new_model = Word2Vec.load('model.bin')
print(new_model)