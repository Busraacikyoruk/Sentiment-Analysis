import nltk
from nltk.corpus import movie_reviews
import random

# Özellik çıkarımı için fonksiyon
def kelime_sıklığına_göre_özellikler(dosya_kelime_listesi):
    return {kelime: True for kelime in dosya_kelime_listesi}

# Yorumları oku ve etiketle
veri = [(kelime_sıklığına_göre_özellikler(movie_reviews.words(dosya)),
         movie_reviews.categories(dosya)[0])
        for dosya in movie_reviews.fileids()]

# Karıştır ve eğitim/test ayır
random.shuffle(veri)
eğitim_seti = veri[:1600]
test_seti = veri[1600:]

# Naive Bayes modelini eğit
classifier = nltk.NaiveBayesClassifier.train(eğitim_seti)

# Başarı oranı
accuracy = nltk.classify.accuracy(classifier, test_seti)
print(f"Model doğruluğu: %{round(accuracy * 100, 2)}")

# Örnek tahmin
örnek_yorum = "This movie was absolutely fantastic and I loved every part of it.".split()
özellikler = kelime_sıklığına_göre_özellikler(örnek_yorum)
tahmin = classifier.classify(özellikler)
print(f"\nTahmin edilen duygu: {tahmin}")
