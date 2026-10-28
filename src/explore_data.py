import pandas as pd
import os

def explore():
    file_path = os.path.join('data', 'aa_dataset-tickets-multi-lang-5-2-50-version.csv')
    
    if not os.path.exists(file_path):
        print(f" Файл не найден по пути: {file_path}")
        print("Убедитесь, что файл лежит в папке data/")
        return

    print("⏳ Загружаем данные (файл большой, может занять пару секунд)...")
    
    # Загружаем данные. Пробуем utf-8, если ошибка - переключаемся на latin-1
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Проблема с кодировкой utf-8, пробуем latin-1...")
        df = pd.read_csv(file_path, encoding='latin-1')
        
    print("Данные успешно загружены!")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок\n")
    
    print("Список всех колонок:")
    print(df.columns.tolist())
    print("-" * 60)
    
    # Проверяем целевую переменную
    if 'queue' in df.columns:
        print("Распределение целевой переменной 'queue' (отделы):")
        print(df['queue'].value_counts())
    else:
        print("Колонка 'queue' не найдена! Проверьте список колонок выше.")
    print("-" * 60)

    # Проверяем пропуски
    print("Пропуски в данных (NaN):")
    print(df.isnull().sum())
    print("-" * 60)

    # Смотрим на примеры текстов
    print("Примеры данных (первые 3 строки):")
    pd.set_option('display.max_colwidth', 150) 
    
    # Ищем колонки с текстом
    cols_to_show = []
    for col in df.columns:
        if any(x in str(col).lower() for x in ['subject', 'description', 'body', 'text', 'queue', 'type', 'topic', 'language']):
            cols_to_show.append(col)
            
    if not cols_to_show:
        cols_to_show = df.columns[:6] # Если не нашли, покажем первые 6
        
    print(df[cols_to_show].head(3))

if __name__ == "__main__":
    explore()