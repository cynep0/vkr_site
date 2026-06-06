from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from ml_pipeline import train_model

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
async def upload_file(
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
        result = train_model(
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


@app.get("/health/")
def health_check():
    return {"status": "ok"}

#uvicorn main:app --reload
#