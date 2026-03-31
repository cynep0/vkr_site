from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from ml_pipeline import train_model

app = FastAPI(title="ML Platform API")

# Разрешаем CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папка для загрузок
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "ML Platform API is running!"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Загружает CSV файл и возвращает результат обучения модели.
    """
    # Проверка расширения
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы!")

    # Сохранение файла
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Обучение модели
        result = train_model(file_path)

        # Удаляем файл после обработки (опционально)
        # os.remove(file_path)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/health/")
def health_check():
    return {"status": "ok"}