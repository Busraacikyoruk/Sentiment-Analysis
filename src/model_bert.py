# model_bert.py

from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import numpy as np
import pandas as pd
import joblib

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

df = pd.read_csv("veriler/temizlenmis_yorumlar.csv")  

def bert_embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

X = np.array([bert_embed(text) for text in df['yorum']])
y = df['etiket']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(clf, "model/bert_logreg_model.pkl")
