import nltk
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')
nltk.download('punkt')

dosyalar = movie_reviews.fileids()

for i in range(3):
    dosya = dosyalar[i]
    kelimeler = movie_reviews.words(dosya)
    duygu = movie_reviews.categories(dosya)[0]

    print(f"\nYorum {i+1} ({duygu}):")
    print(" ".join(kelimeler[:50]))
