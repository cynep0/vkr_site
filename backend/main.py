from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from ml_pipeline import train_logistic_regression, train_linear_regression,train_random_forest, train_kmeans

app = FastAPI(title="ML Platform API")

# Разрешаем CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ML Platform API is running!"}


@app.post("/logistic_regression/")
async def train_logistic_reg(
    file: UploadFile = File(...),
    target_column: str = Form(...),  # Обязательно
    feature_columns: str = Form(...),
    C: float = Form(1.0),
    penalty: str = Form("l2"),
    max_iter: int = Form(1000),
    test_size: float = Form(0.2),
    class_weight: str = Form("None")
):
    # Проверка расширения
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы!")

    # Парсим список признаков из JSON
    try:
        features = json.loads(feature_columns)
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("feature_columns должен быть непустым списком")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Неверный формат feature_columns")

    # попытка обучить модель
    try:
        result = train_logistic_regression(
            file.file,
            target_column=target_column,
            feature_columns=features,
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            test_size=test_size,
            class_weight=class_weight
        )
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/linear_regression/")
async def train_linear_reg(
    file: UploadFile = File(...),
    target_column: str = Form(...),  # Обязательно
    feature_columns: str = Form(...),
    alpha: float = Form(1.0),
    penalty: str = Form("l2"),
    max_iter: int = Form(1000),
    test_size: float = Form(0.2),
    fit_intercept: bool = Form(True)
):
    # Проверка расширения
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы!")

    # Парсим список признаков из JSON
    try:
        features = json.loads(feature_columns)
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("feature_columns должен быть непустым списком")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Неверный формат feature_columns")

    # попытка обучить модель
    try:
        result = train_linear_regression(
            file.file,
            target_column=target_column,
            feature_columns=features,
            model_type=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            test_size=test_size
        )
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.post("/random_forest/")
async def train_random_forest_endpoint(
        file: UploadFile = File(...),
        target_column: str = Form(...),
        feature_columns: str = Form(...),
        task_type: str = Form("classification"),  # "classification" или "regression"
        n_estimators: int = Form(100, ge=10, le=500),
        max_depth: int = Form(10, ge=1, le=100),
        min_samples_split: int = Form(2, ge=2, le=20),
        max_features: str = Form("sqrt"),
        test_size: float = Form(0.2, ge=0.1, le=0.5)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы!")

    try:
        features = json.loads(feature_columns)
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("feature_columns должен быть непустым списком")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Неверный формат feature_columns")

    if target_column in features:
        raise HTTPException(status_code=400, detail="Целевая переменная не должна быть в признаках!")

    try:
        result = train_random_forest(
            file.file,
            target_column=target_column,
            feature_columns=features,
            task_type=task_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            test_size=test_size
        )
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/kmeans/")
async def train_kmeans_endpoint(
        file: UploadFile = File(...),
        feature_columns: str = Form(...),
        n_clusters: int = Form(3, ge=2, le=20),
        init: str = Form("k-means++"),
        max_iter: int = Form(300, ge=100, le=1000),
        n_init: int = Form(10, ge=1, le=50)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы!")

    try:
        features = json.loads(feature_columns)
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("feature_columns должен быть непустым списком")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Неверный формат feature_columns")

    try:
        result = train_kmeans(
            file.file,
            feature_columns=features,
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init
        )
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/health/")
def health_check():
    return {"status": "ok"}

#uvicorn main:app --reload
#