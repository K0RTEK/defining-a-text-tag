import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import joblib
# Загрузка данных
data = pd.read_csv('train_data.csv')
# Предобработка текстов
data['text'] = data['text'].fillna('').apply(lambda x: re.sub(r'[^a-zA-Zа-яА-Я]', ' ', x))

# Разделение на признаки и целевую переменную
X = data['text']
y = data['tag']
# Разбиение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание пайплайна
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=nltk.corpus.stopwords.words('russian'))),
    ('clf', MultinomialNB())
])

# Подбор гиперпараметров
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__max_df': [0.5, 0.75, 1.0],
    'clf__alpha': [0.1, 0.5, 1.0]
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Оценка качества модели на тестовой выборке
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on test set: {:.2f}'.format(accuracy))

# # Оценка качества модели на отложенной выборке
# data_holdout = pd.read_csv('holdout_data.csv')
# data_holdout['text'] = data_holdout['text'].apply(lambda x: re.sub(r'[^a-zA-Zа-яА-Я]', ' ', x))
# X_holdout = data_holdout['text']
# y_holdout = data_holdout['tag']
# y_pred_holdout = grid_search.predict(X_holdout)
# accuracy_holdout = accuracy_score(y_holdout, y_pred_holdout)
# print('Accuracy on holdout set: {:.2f}'.format(accuracy_holdout))


joblib.dump(grid_search.best_estimator_, 'text_model.pkl')
print('Best parameters: {}'.format(grid_search.best_params_))


