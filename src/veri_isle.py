import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("C:/Users/aciky/Desktop/Sentiment-Analysis/ham_yorumlar.csv")

stop_words = set(stopwords.words('turkish'))

def temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    kelimeler = metin.split()
    kelimeler = [k for k in kelimeler if k not in stop_words]
    return " ".join(kelimeler)

df['temiz_yorum'] = df['yorum'].apply(temizle)

df[['temiz_yorum', 'etiket']].to_csv("C:/Users/aciky/Desktop/Sentiment-Analysis/temizlenmis_yorumlar.csv", index=False)

print("✅ Temizlenmiş yorumlar dosyası başarıyla oluşturuldu.")
