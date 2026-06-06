import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import numpy as np
from sklearn.preprocessing import LabelEncoder


def train_model(file_object,
                target_column: str,
                feature_columns: list,
                C: float = 1.0,
                penalty: str = "l2",
                max_iter: int = 1000,
                test_size: float = 0.2,
                class_weight: str = "None" ) -> dict:
    """
    Обучает модель на загруженных данных и возвращает метрики.
    """
    # Чтение данных
    file_object.seek(0)
    df = pd.read_csv(file_object)

    # Простая предобработка: удаляем строки с пропусками
    df = df.dropna()

    # === Валидация столбцов ===
    available_columns = list(df.columns)

    if target_column not in available_columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных")

    for col in feature_columns:
        if col not in available_columns:
            raise ValueError(f"Признак '{col}' не найден в данных")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

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
        X, y, test_size=test_size, random_state=42
    )

    # Определяем l1_ratio на основе выбранного penalty
    if penalty == "l2":
        l1_ratio = 0
    elif penalty == "l1":
        l1_ratio = 1
    else:  # none
        l1_ratio = None
        C = np.inf  # Отключаем регуляризацию

    # Обучение модели
    model = LogisticRegression(
        C=C if penalty != "none" else np.inf,
        l1_ratio=l1_ratio if penalty != "none" else None,
        solver="lbfgs" if penalty != "l1" else "saga",
        max_iter=max_iter,
        class_weight=None if class_weight=="None" else class_weight)
    model.fit(X_train, y_train)

    # Предсказание и оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # === 2. Дополнительные метрики (для классификации) ===
    # average='binary' для бинарной, 'macro' для мультиклассовой
    avg_strategy = 'binary' if len(np.unique(y)) == 2 else 'macro'

    precision = precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)

    # === 3. ROC-AUC (только для бинарной классификации) ===
    roc_auc = None
    if len(np.unique(y)) == 2:
        try:
            # Нужны вероятности, а не классы
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None  # Если не получилось посчитать

    # === 4. Матрица ошибок ===
    cm = confusion_matrix(y_test, y_pred)

    # === 5. Коэффициенты модели (важность признаков) ===
    # Для LogisticRegression: coef_ показывает вес каждого признака
    feature_importance = {}
    if hasattr(model, 'coef_'):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        for name, value in zip(feature_columns, coef):
            feature_importance[name] = round(float(value), 4)

    # === 6. Информация о сходимости ===
    convergence_info = {
        "n_iter": getattr(model, 'n_iter_', [None])[0] if hasattr(model, 'n_iter_') else None,
        "converged": getattr(model, 'converged_', True) if hasattr(model, 'converged_') else True
    }

    # логи в консоль:
    print("Модель обучена!")
    print(f"Сила регуляризации = {C}")
    print(f"Тип регуляризации = {penalty}")
    print(f"Количество итераций = {max_iter}")
    print(f"Доля тестовой выборки= {test_size}")
    print(f"Accuracy: {accuracy}")

    print("--------")
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1_score: ", f1)
    print("roc_auc: ", roc_auc)
    print(cm.tolist())
    print([int(x) if isinstance(x, (np.integer, np.floating)) else str(x)
                                    for x in np.unique(y)])
    print({k: float(v) for k, v in feature_importance.items()})
    print({
            "n_iter": int(convergence_info["n_iter"]) if convergence_info["n_iter"] is not None else None,
            "converged": bool(convergence_info["converged"])
        })

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,

        "train_size": len(X_train),
        "test_size": len(X_test),

        "features": feature_columns,
        "target_column": target_column,

        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [int(x) if isinstance(x, (np.integer, np.floating)) else str(x)
                                    for x in np.unique(y)],  # Классы: [0, 1] или ['No', 'Yes']

        # Важность признаков
        "feature_importance":  {k: float(v) for k, v in feature_importance.items()},

        # Информация о сходимости
        "convergence": {
            "n_iter": int(convergence_info["n_iter"]) if convergence_info["n_iter"] is not None else None,
            "converged": bool(convergence_info["converged"])
        },

        "message": "Модель успешно обучена!"
    }