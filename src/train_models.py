import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(text):
    """Очистка текста: нижний регистр, только буквы и пробелы."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_evaluate():
    file_path = os.path.join('data', 'aa_dataset-tickets-multi-lang-5-2-50-version.csv')
    print("Загрузка данных...")
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Оставляем только английский язык
    df = df[df['language'] == 'en'].copy()
    
    print("Предобработка текста...")
    df['subject_clean'] = df['subject'].apply(clean_text)
    df['body_clean'] = df['body'].apply(clean_text)
    
    # Объединяем тему и описание
    df['text'] = df['subject_clean'] + " " + df['body_clean']
    
    X = df['text']
    y = df['queue']

    # Разделение на Train и Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Настройка пайплайна (лучшая конфигурация)...")
    # Возвращаемся к лучшей конфигурации: Word TF-IDF (1,2) + LinearSVC с балансировкой
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=2000))
    ])

    # Подбор гиперпараметров (C)
    param_grid = {
        'clf__C': [1.0, 5.0, 10.0, 20.0]
    }

    print("Запуск GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"\nЛучшие параметры: {grid_search.best_params_}")
    print(f"Лучший Macro F1 на кросс-валидации: {grid_search.best_score_:.4f}")

    # Оценка на тестовой выборке
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ФИНАЛЬНОЙ МОДЕЛИ НА ТЕСТЕ:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("="*50)
    
    print("\nДетальный отчет:")
    print(classification_report(y_test, y_pred))

    # Сохранение финальной модели
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'ticket_classifier_final.pkl')
    joblib.dump(best_model, model_path)
    print(f"Модель сохранена: {model_path}")

    # Сохранение метрик для отчета
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/final_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f"Best params: {grid_search.best_params_}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()