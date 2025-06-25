import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
bert_model = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased").to("cpu")

with open("src/duygu_modeli.pkl", "rb") as dosya:
    model = pickle.load(dosya)

def temizle(metin):
    stop_words = set(stopwords.words("turkish"))
    kelimeler = word_tokenize(metin.lower())
    temiz_kelimeler = [k for k in kelimeler if k.isalpha() and k not in stop_words]
    return " ".join(temiz_kelimeler)

st.title("🎬 Film Yorumları Üzerine Duygu Analizi")
st.write("BERT tabanlı sınıflandırıcı ile yorumun olumlu mu olumsuz mu olduğunu tahmin edin.")

yorum = st.text_area("Yorumunuzu yazın:", "")

if st.button("Analiz Et"):
    if not yorum.strip():
        st.warning("⚠️ Lütfen boş olmayan bir yorum girin.")
        st.stop()
    else:
        try:
            temiz_yorum = temizle(yorum)

            inputs = tokenizer(
                temiz_yorum,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            if inputs["input_ids"].nelement() == 0:
                st.error("⚠️ Geçerli bir giriş üretilemedi. Lütfen daha anlamlı bir yorum girin.")
                st.stop()
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = bert_model(**inputs)

            if outputs.last_hidden_state is None:
                st.error("BERT modeli boş çıktı üretti.")
                st.stop()

            vektor = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            tahmin = model.predict(vektor)[0]

            if tahmin == "pos":
                st.success("💚 Bu yorum **olumlu** olarak değerlendirildi.")
            else:
                st.error("❤️‍🩹 Bu yorum **olumsuz** olarak değerlendirildi.")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
