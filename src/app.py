import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
noktalama = set(string.punctuation)

def yorum_temizle(metin):
    # Küçük harfe çevir
    metin = metin.lower()
    # Noktalama işaretlerini çıkar
    metin = ''.join(ch for ch in metin if ch not in noktalama)
    # Kelimelere ayır
    kelimeler = metin.split()
    # Stopword’leri çıkar
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return temiz_kelimeler

import streamlit as st
import nltk
from nltk.corpus import movie_reviews
import random

# Özellik çıkarımı fonksiyonu
def kelime_sıklığına_göre_özellikler(dosya_kelime_listesi):
    return {kelime: True for kelime in dosya_kelime_listesi}

# Model eğitimi (basit, hafif)
@st.cache(allow_output_mutation=True)
def model_egit():
    nltk.download('movie_reviews')
    veri = [(kelime_sıklığına_göre_özellikler(movie_reviews.words(dosya)),
             movie_reviews.categories(dosya)[0])
            for dosya in movie_reviews.fileids()]
    random.shuffle(veri)
    eğitim_seti = veri[:1600]
    test_seti = veri[1600:]
    classifier = nltk.NaiveBayesClassifier.train(eğitim_seti)
    return classifier

classifier = model_egit()

st.title("Duygu Analizi Web Uygulaması")

yorum = st.text_area("Bir film yorumu yazın:")

if st.button("Tahmin Et"):
    if yorum.strip() == "":
        st.warning("Lütfen bir yorum girin!")
    else:
        temiz_kelime_listesi = yorum_temizle(yorum)
        if not temiz_kelime_listesi:
            st.warning("Yorumunuz anlamlı kelimeler içermiyor.")
        else:
            özellikler = kelime_sıklığına_göre_özellikler(temiz_kelime_listesi)
            tahmin = classifier.classify(özellikler)
            st.success(f"Bu yorumun duygusu: **{tahmin.upper()}**")

