import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score
)
import numpy as np
from sklearn.preprocessing import LabelEncoder


def train_logistic_regression(file_object,
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


def train_linear_regression(file_object,
                            target_column: str,
                            feature_columns: list,
                            model_type: str = "ridge",
                            alpha: float = 1.0,
                            fit_intercept: bool = True,
                            max_iter: int = 1000,
                            test_size: float = 0.2) -> dict:
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

    # === Проверка: целевая переменная должна быть числовой ===
    if y.dtype == 'object':
        raise ValueError(f"Для линейной регрессии целевая переменная '{target_column}' должна быть числовой!")

    # Кодирование категориальных признаков в X
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Выбрана модель: {model_type} (alpha={alpha})")
    if model_type == "none":
        # Обычная линейная регрессия без регуляризации
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

    elif model_type == "l2":
        # Ridge: L2-регуляризация
        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)

    elif model_type == "l1":
        # Lasso: L1-регуляризация
        model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            random_state=42
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    # === Предсказание и метрики ===
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # === Коэффициенты модели ===
    feature_importance = {}
    if hasattr(model, 'coef_'):
        coef = model.coef_
        for name, value in zip(feature_columns, coef):
            feature_importance[name] = round(float(value), 4)

    # === Данные для графиков ===
    plot_data = []
    for i in range(min(len(y_test), 100)):
        plot_data.append({
            "actual": float(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]),
            "predicted": float(y_pred[i])
        })

    residuals = [float((y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]) - y_pred[i])
                 for i in range(min(len(y_test), 100))]

    # === Количество обнулённых коэффициентов (для Lasso) ===
    n_zero_coef = 0
    if model_type == "lasso" and hasattr(model, 'coef_'):
        n_zero_coef = int(np.sum(np.abs(model.coef_) < 1e-10))

    print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}")
    if model_type == "lasso":
        print(f"Lasso: {n_zero_coef} из {len(feature_columns)} коэффициентов обнулено")

    # === 6. Информация о сходимости ===
    convergence_info = {
        "n_iter": _safe_get_n_iter(model),
        "converged": getattr(model, 'converged_', True) if hasattr(model, 'converged_') else True
    }

    # === Возврат результатов ===
    result = {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),

        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),

        "features": feature_columns,
        "model_type": model_type,
        "alpha_used": alpha if model_type != "linear" else None,

        "feature_importance": {k: float(v) for k, v in feature_importance.items()},

        "predicted_vs_actual": plot_data,
        "residuals": residuals,

        # Информация о сходимости
        "convergence": {
            "n_iter": int(convergence_info["n_iter"]) if convergence_info["n_iter"] is not None else None,
            "converged": bool(convergence_info["converged"])
        },
        "message": f"Модель {model_type.upper()} успешно обучена!",

    }

    return result


def train_random_forest(
        file_object,
        target_column: str,
        feature_columns: list,
        task_type: str = "classification",  # "classification" или "regression"
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        test_size: float = 0.2
) -> dict:
    """
    Обучает модель Random Forest для классификации или регрессии.
    """

    file_object.seek(0)
    df = pd.read_csv(file_object)
    df = df.dropna()

    if len(df) < 10:
        raise ValueError("Слишком мало данных для обучения (минимум 10 строк)")

    # === Выбор столбцов ===
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # === Кодирование категориальных признаков ===
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # === Кодирование target для классификации ===
    is_classification = task_type == "classification"
    target_encoder = None

    if is_classification:
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
    else:
        # Для регрессии target должен быть числовым
        if y.dtype == 'object':
            try:
                y = pd.to_numeric(y)
            except:
                raise ValueError(f"Для регрессии целевая переменная '{target_column}' должна быть числовой!")

    # === Разделение на train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # === Выбор и обучение модели ===
    print(f"Random Forest: {task_type}, n_estimators={n_estimators}, max_depth={max_depth}")

    if max_features == "1.0":
        max_features = 1.0  # Строка → float (100% признаков)

    if max_features == "0.3":
        max_features = 0.3  # Строка → float (100% признаков)

    if is_classification:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )

    model.fit(X_train, y_train)

    # === Предсказание ===
    y_pred = model.predict(X_test)

    # === Метрики ===
    if is_classification:
        # Метрики классификации
        accuracy = accuracy_score(y_test, y_pred)
        avg_strategy = 'binary' if len(np.unique(y)) == 2 else 'macro'
        precision = precision_score(y_test, y_pred, average=avg_strategy, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_strategy, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)

        # ROC-AUC только для бинарной классификации
        roc_auc = None
        if len(np.unique(y)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                pass

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": [int(x) for x in np.unique(y)]
        }
    else:
        # Метрики регрессии
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4)
        }

    # === Важность признаков ===
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, value in zip(feature_columns, importances):
            feature_importance[name] = round(float(value), 4)

    # === Данные для графиков ===
    plot_data = []
    for i in range(min(len(y_test), 100)):
        plot_data.append({
            "actual": float(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]),
            "predicted": float(y_pred[i])
        })

    residuals = [float((y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]) - y_pred[i])
                 for i in range(min(len(y_test), 100))]

    # === Статистика деревьев ===
    tree_stats = {
        "n_trees": n_estimators,
        "max_depth_actual": int(model.estimators_[0].tree_.max_depth) if hasattr(model, 'estimators_') else None,
    }

    print(f"{'Accuracy' if is_classification else 'R²'}: {metrics.get('accuracy', metrics.get('r2_score')):.4f}")

    # === Возврат результатов ===
    result = {
        **metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "feature_importance": {k: float(v) for k, v in feature_importance.items()},
        "tree_stats": tree_stats,
        "predicted_vs_actual": plot_data,
        "residuals": residuals,
        "message": f"Random Forest ({task_type}) успешно обучен!",
    }
    return result

def train_kmeans(
    file_object,
    feature_columns: list,
    n_clusters: int = 3,
    init: str = "k-means++",
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42
) -> dict:
    """
    Обучает модель K-Means для кластеризации данных.
    """
    file_object.seek(0)
    df = pd.read_csv(file_object)
    df = df.dropna()

    if len(df) < n_clusters * 5:
        raise ValueError(f"Недостаточно данных для {n_clusters} кластеров (минимум {n_clusters * 5} строк)")

    # === Используем только выбранные признаки (НЕТ целевой переменной!) ===
    X = df[feature_columns].copy()

    # === Кодирование категориальных признаков ===
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # === Масштабирование для K-Means) ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Обучение модели ===
    print(f"✅ K-Means: n_clusters={n_clusters}, init={init}")

    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        algorithm="lloyd"
    )
    model.fit(X_scaled)

    # === Предсказание кластеров ===
    cluster_labels = model.predict(X_scaled)

    # === Метрики кластеризации ===
    inertia = model.inertia_  # Within-cluster sum of squares
    silhouette = silhouette_score(X_scaled, cluster_labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels) if n_clusters > 1 else float('inf')

    # === Центроиды кластеров ===
    centroids = {}
    for i, centroid in enumerate(model.cluster_centers_):
        centroids[f"Cluster_{i}"] = {
            feature_columns[j]: round(float(centroid[j]), 4)
            for j in range(len(feature_columns))
        }

    # === Распределение объектов по кластерам ===
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
    cluster_distribution = {f"Cluster_{k}": int(v) for k, v in cluster_counts.items()}

    # === Данные для визуализации (PCA для 2D) ===
    # Если признаков > 2, используем PCA для проекции на 2D
    if len(feature_columns) > 2:
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X_scaled)
        explained_variance = round(float(pca.explained_variance_ratio_.sum() * 100), 2)
    else:
        X_2d = X_scaled[:, :2] if X_scaled.shape[1] >= 2 else np.hstack([X_scaled, np.zeros((len(X_scaled), 1))])
        explained_variance = 100.0

    # === Точки для графика ===
    scatter_data = []
    for i in range(min(len(X_2d), 500)):  # Ограничиваем для производительности
        scatter_data.append({
            "x": float(X_2d[i][0]),
            "y": float(X_2d[i][1]),
            "cluster": int(cluster_labels[i])
        })

    # === Центроиды для графика (тоже через PCA) ===
    if len(feature_columns) > 2:
        centroids_2d = pca.transform(model.cluster_centers_)
    else:
        centroids_2d = model.cluster_centers_[:, :2]

    centroid_plot_data = []
    for i, (x, y) in enumerate(centroids_2d):
        centroid_plot_data.append({
            "x": float(x),
            "y": float(y),
            "cluster": i
        })

    print(f"✅ Inertia: {inertia:.2f}, Silhouette: {silhouette:.4f}")
    print(f"✅ Распределение: {cluster_distribution}")

    # === Возврат результатов ===
    result = {
        # Метрики
        "inertia": round(float(inertia), 4),
        "silhouette_score": round(float(silhouette), 4),
        "davies_bouldin_score": round(float(davies_bouldin), 4),

        # Информация о модели
        "n_clusters": n_clusters,
        "n_samples": int(len(X)),
        "features": feature_columns,

        # Центроиды
        "centroids": centroids,
        "cluster_distribution": cluster_distribution,

        # Данные для графиков
        "scatter_data": scatter_data,
        "centroid_data": centroid_plot_data,
        "pca_explained_variance": explained_variance,

        "message": f"K-Means успешно обучена!",
    }
    return result

def _safe_get_n_iter(model):
    """Безопасно извлекает количество итераций, если атрибут существует."""
    n_iter_raw = getattr(model, 'n_iter_', None)
    if n_iter_raw is None:
        return None
    # Если это массив/список/кортеж — берём первый элемент
    if isinstance(n_iter_raw, (np.ndarray, list, tuple)):
        return int(n_iter_raw[0]) if len(n_iter_raw) > 0 else None
    # Если скаляр (int/float) — возвращаем как есть
    return int(n_iter_raw) if isinstance(n_iter_raw, (int, float)) else None