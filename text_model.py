from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords

# Загрузка данных
data = pd.read_csv('train_data.csv')

# Предварительная обработка данных
stop_words = stopwords.words('russian')
vect = CountVectorizer(stop_words=stop_words)
X = vect.fit_transform(data['text'].values.astype('U'))
y = data['tag']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Обучение модели
model = MultinomialNB()
model.fit(X_train, y_train)

# Оценка качества модели на тестовой выборке
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Применение модели для определения тега текста
text = ''
text_processed = vect.transform([text])
tag = model.predict(text_processed)[0]
print('Tag:', tag)

with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)
