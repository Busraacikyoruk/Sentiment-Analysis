import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
import string
import random

nltk.download('stopwords')
nltk.download('punkt')

# Stopwords ve noktalama işaretleri
stop_words = set(stopwords.words("english"))
noktalama = set(string.punctuation)

def temizle(kelimeler):
    temiz = [kelime.lower() for kelime in kelimeler]
    temiz = [kelime for kelime in temiz if kelime not in stop_words and kelime not in noktalama]
    return temiz

# Veri listeleri [(kelimeler, duygu)]
veri = []
for dosya in movie_reviews.fileids():
    kelimeler = movie_reviews.words(dosya)
    temiz_kelimeler = temizle(kelimeler)
    etiket = movie_reviews.categories(dosya)[0]
    veri.append((temiz_kelimeler, etiket))

# Karıştır
random.shuffle(veri)

# İlk 3 örneği göster
for i in range(3):
    print(f"\nYorum {i+1} - Etiket: {veri[i][1]}")
    print(" ".join(veri[i][0][:30]))
