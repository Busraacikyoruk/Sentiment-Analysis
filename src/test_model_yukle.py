import os
import pickle

print("Aktif klasör:", os.getcwd())

if os.path.exists("duygu_modeli.pkl"):
    print("✅ Model dosyası bulundu.")
    with open("duygu_modeli.pkl", "rb") as dosya:
        model = pickle.load(dosya)
        print("✅ Model başarıyla yüklendi.")
else:
    print("❌ Model dosyası bulunamadı.")
