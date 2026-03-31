document.addEventListener('DOMContentLoaded', () => {

    // 1. Получение элементов
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const resultBox = document.getElementById('resultBox');
    const errorBox = document.getElementById('error');
    
    const API_URL = 'http://127.0.0.1:8000/upload/';
    
    // Клик по зоне загрузки
    dropZone.addEventListener('click', () => fileInput.click());
    
    // Выбор файла через диалог
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });
    
    // Drag-and-Drop события
    // Файл над зоной
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    // Файл ушёл с зоны
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    // Файл сброшен
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });
    
    // Загрузка файла на сервер
    async function uploadFile(file) {
        // Проверка расширения
        if (!file.name.endsWith('.csv')) {
            showError('Пожалуйста, загрузите CSV файл!');
            return;
        }
        
        // Сброс предыдущих результатов
        resultBox.classList.remove('show');
        errorBox.classList.remove('show');
        loading.classList.add('show');
        
        // Подготовка данных
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Ошибка загрузки');
            }
            
            // Отображение результатов
            displayResults(data);
            
        } catch (error) {
            showError(error.message);
        } finally {
            loading.classList.remove('show');
        }
    }
    
    // Отображение результатов
    function displayResults(data) {
        document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
        document.getElementById('trainSize').textContent = data.train_size;
        document.getElementById('testSize').textContent = data.test_size;
        document.getElementById('features').textContent = data.features.length;
        document.getElementById('message').textContent = data.message;
        
        resultBox.classList.add('show');
    }
    
    // Отображение ошибки
    function showError(message) {
        errorBox.textContent = '❌ ' + message;
        errorBox.classList.add('show');
    }
});