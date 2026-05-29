# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from google_play_scraper import reviews_all

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# SCRAPING DATA REVIEW BRIMO
result = reviews_all(
    'id.co.bri.brimo',
    sleep_milliseconds=0,
    lang='id',
    country='id'
)

# UBAH KE DATAFRAME
data = pd.DataFrame(result)

data.head()

# AMBIL 3000 DATA SAJA
data = data.head(3000)

print("Jumlah data:", len(data))

# AMBIL KOLOM YANG DIGUNAKAN
data = data[['content', 'score', 'at']]

data.head()

# CEK INFO DATA
data.info()

# CEK MISSING VALUE
data.isnull().sum()

# PREPROCESSING

# CASE FOLDING & CLEANING
def cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['clean'] = data['content'].astype(str).apply(cleaning)

# STOPWORD REMOVAL
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

# stopword tambahan
custom_stopword = [
    'nya', 'yg', 'aja', 'udah', 'apk',
    'banget', 'ga', 'gak', 'sih',
    'nih', 'deh', 'dong', 'kalo',
    'kalau', 'buat', 'jadi', 'lebih',
    'sangat'
]

def remove_custom_stopword(text):
    text = stopword.remove(text)

    words = text.split()

    words = [word for word in words if word not in custom_stopword]

    return ' '.join(words)

data['stopword'] = data['clean'].apply(remove_custom_stopword)

# STEMMING
factory = StemmerFactory()
stemmer = factory.create_stemmer()

data['stem'] = data['stopword'].apply(stemmer.stem)

# PERBANDINGAN SEBELUM DAN SESUDAH PREPROCESSING
data[['content', 'clean', 'stem']].head()

# LABEL SENTIMEN
def sentiment(score):
    if score >= 4:
        return 'positif'
    elif score <= 2:
        return 'negatif'
    else:
        return 'netral'

data['sentiment'] = data['score'].apply(sentiment)

# DISTRIBUSI RATING
plt.figure(figsize=(6,4))

data['score'].value_counts().sort_index().plot(kind='bar')

plt.title('Distribusi Rating BRImo')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()

# DISTRIBUSI SENTIMEN
plt.figure(figsize=(6,6))

data['sentiment'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title('Distribusi Sentimen')
plt.ylabel('')
plt.show()

# WORDCLOUD
text = ' '.join(data['stem'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# TOP 10 KATA TERBANYAK
from collections import Counter

all_words = ' '.join(data['stem']).split()

counter = Counter(all_words)

top_words = pd.DataFrame(
    counter.most_common(10),
    columns=['Kata', 'Jumlah']
)

plt.figure(figsize=(8,5))

sns.barplot(
    x='Jumlah',
    y='Kata',
    data=top_words
)

plt.title('Top 10 Kata Terbanyak')
plt.show()

data['length'] = data['content'].astype(str).apply(len)

plt.figure(figsize=(8,5))

sns.histplot(data['length'])

plt.title('Distribusi Panjang Review')
plt.show()

# HAPUS DATA NETRAL
data_ml = data[data['sentiment'] != 'netral']

# TF-IDF
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data_ml['stem'])

y = data_ml['sentiment']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAINING MODEL
model = MultinomialNB()

model.fit(X_train, y_train)

# PREDIKSI
y_pred = model.predict(X_test)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# CLASSIFICATION REPORT
print(classification_report(y_test, y_pred))
