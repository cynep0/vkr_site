import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os


def train_model(file_path: str) -> dict:
    """
    Обучает модель на загруженных данных и возвращает метрики.
    """
    # Чтение данных
    df = pd.read_csv(file_path)

    # Простая предобработка: удаляем строки с пропусками
    df = df.dropna()

    # Разделение на признаки и целевую переменную
    # Предполагаем, что последний столбец - целевая переменная
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Если целевая переменная текстовая - кодируем её
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Кодирование категориальных признаков в X
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение модели
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Предсказание и оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # В конце функции train_model, перед return:
    print(f"✅ Accuracy: {accuracy}")
    print(f"✅ Предсказания: {y_pred[:5]}")  # Первые 5 предсказаний

    return {
        "accuracy": round(accuracy, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": list(X.columns),
        "message": "Модель успешно обучена!"
    }