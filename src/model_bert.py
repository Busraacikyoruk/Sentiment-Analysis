# model_bert.py

from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np
import pandas as pd
import joblib

# BERT modeli ve tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

# Veri setini oku (veri_yukle.py tarafından kaydedilmişse)
df = pd.read_csv("veriler/temizlenmis_yorumlar.csv")  # örnek dosya adı, sende neyse onu yaz

# Fonksiyon: bir metni BERT embedding'e çevir
def bert_embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token'ı ile temsil edilen vektörü al
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# Tüm veriler için embedding üret
X = np.array([bert_embed(text) for text in df['yorum']])
y = df['etiket']  # Etiket kolonu "pozitif"/"negatif" olmalı

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit (örnek: lojistik regresyon)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Tahmin yap
y_pred = clf.predict(X_test)

# Raporla
print(classification_report(y_test, y_pred))

# Eğitilen modeli kaydet
joblib.dump(clf, "model/bert_logreg_model.pkl")
