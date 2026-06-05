document.addEventListener('DOMContentLoaded', () => {

    // 1. Получение элементов
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const loading = document.getElementById('loading');
    const columnSelection = document.getElementById('columnSelection');
    const trainButton = document.getElementById('trainButton');
    const resultBox = document.getElementById('resultBox');
    const errorBox = document.getElementById('error');
    
    const API_URL = 'http://127.0.0.1:8000/upload/';

    // === Глобальная переменная для хранения файла ===
    let selectedFile = null;

    let selectedColumns = {
        target: null,
        features: []
    };
    
    // Клик по зоне загрузки
    dropZone.addEventListener('click', () => fileInput.click());
    
    // Выбор файла через диалог
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
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
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        // Проверка расширения
        if (!file.name.endsWith('.csv')) {
            showError('Пожалуйста, загрузите CSV файл!');
            return;
        }
        
        // Сохраняем файл в переменную
        selectedFile = file;
        
        // Показываем информацию о файле
        fileName.textContent = file.name;
        fileInfo.style.display = 'block';

        // Показываем выбор столбцов
        columnSelection.classList.add('show');
        loadColumnNames(file);
        
        // Скрываем старые результаты
        resultBox.classList.remove('show');
        errorBox.classList.remove('show');
        
        // Зона загрузки становится менее заметной
        dropZone.style.opacity = '0.6';
        dropZone.style.pointerEvents = 'none';
    }

    async function loadColumnNames(file) {
        const targetSelect = document.getElementById('targetColumn');
        const featuresList = document.getElementById('featuresList');
    
        // Читаем первую строку CSV для получения имён столбцов
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const firstLine = text.split('\n')[0];
            const columns = firstLine.split(',').map(col => col.trim().replace(/"/g, ''));

            // Очищаем существующие опции
            targetSelect.innerHTML = '<option value="">-- Выберите столбец --</option>';
            featuresList.innerHTML = '';

            // Заполняем селект целевой переменной
            columns.forEach((col, index) => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                targetSelect.appendChild(option);

                // Заполняем список признаков (чекбоксы)
                const checkboxDiv = document.createElement('div');
                checkboxDiv.style.marginBottom = '8px';
                checkboxDiv.innerHTML = `
                    <label class="checkBox-label">
                        ${col}
                        <input type="checkbox" class="feature-checkbox" value="${col}">
                    </label>
                `;
                featuresList.appendChild(checkboxDiv);
            });

            // Добавляем обработчик изменения целевой переменной
            targetSelect.addEventListener('change', validateColumnSelection);

            // Добавляем обработчики на чекбоксы
            document.querySelectorAll('.feature-checkbox').forEach(cb => {
                cb.addEventListener('change', validateColumnSelection);
            });
        };
        reader.readAsText(file);
    }

    function validateColumnSelection() {
        const targetColumn = document.getElementById('targetColumn').value;
        const featureCheckboxes = document.querySelectorAll('.feature-checkbox:checked');
        const features = Array.from(featureCheckboxes).map(cb => cb.value);
        const warningDiv = document.getElementById('columnWarning');
        const warningText = document.getElementById('warningText');
        
        let warnings = [];
        
        // Проверка: выбрана ли целевая переменная
        if (!targetColumn) {
            warnings.push('Выберите целевую переменную');
        }
        
        // Проверка: выбраны ли признаки
        if (features.length === 0) {
            warnings.push('Выберите минимум 1 признак');
        }
        
        // Проверка: не совпадает ли целевая с признаком
        if (targetColumn && features.includes(targetColumn)) {
            warnings.push('Целевая переменная не должна быть в признаках!');
        }
        
        // Показываем предупреждения
        if (warnings.length > 0) {
            warningText.textContent = warnings.join('. ');
            warningDiv.style.display = 'block';
        } else {
            warningDiv.style.display = 'none';
        }
        
        return warnings.length === 0;
    }

    function validateParams() {
        let isValid = true;
        
        // === Валидация C ===
        const cInput = document.getElementById('paramC');
        const cError = document.getElementById('cError');
        const cValue = parseFloat(cInput.value);
        
        if (isNaN(cValue) || cValue < 0.01 || cValue > 1000) {
            cError.style.display = 'block';
            cInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            cError.style.display = 'none';
            cInput.style.borderColor = '#ddd';
        }
        
        // === Валидация max_iter ===
        const maxIterInput = document.getElementById('paramMaxIter');
        const maxIterError = document.getElementById('maxIterError');
        const maxIterValue = parseInt(maxIterInput.value);
        
        if (isNaN(maxIterValue) || maxIterValue < 100 || maxIterValue > 10000) {
            maxIterError.style.display = 'block';
            maxIterInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            maxIterError.style.display = 'none';
            maxIterInput.style.borderColor = '#ddd';
        }
        
        return isValid;
    }

    // Обучение модели (по кнопке)
    window.trainModel = async function() {
        // Сброс интерфейса
        resultBox.classList.remove('show');
        errorBox.classList.remove('show');

        if (!selectedFile) {
            showError('Сначала загрузите файл!');
            return;
        }

        if (!validateParams()) {
            showError('Исправьте ошибки в параметрах модели!');
            return;
        }

        if (!validateColumnSelection()) {
            showError('Выберете корректные столбцы!');
            return;
        }

        loading.classList.add('show');
        trainButton.disabled = true;
        trainButton.textContent = '⏳ Обучение...';

        // === Подготовка FormData ===
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // === Добавляем гиперпараметры ===
        formData.append('C', document.getElementById('paramC').value);
        formData.append('penalty', document.getElementById('paramPenalty').value);
        formData.append('max_iter', document.getElementById('paramMaxIter').value);
        formData.append('test_size', document.getElementById('paramTestSize').value);
        

        // Сохраняем выбор столбцов
        selectedColumns.target = document.getElementById('targetColumn').value;
        selectedColumns.features = Array.from(document.querySelectorAll('.feature-checkbox:checked')).map(cb => cb.value);
        formData.append('target_column', selectedColumns.target);
        formData.append('feature_columns', JSON.stringify(selectedColumns.features));


        try {
            // === Отправка запроса ===
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Ошибка обучения');
            }
            
            // === Отображение результатов ===
            displayResults(data);
            
        } catch (error) {
            showError(error.message);
        } finally {
            loading.classList.remove('show');
            trainButton.disabled = false;
            trainButton.textContent = 'Обучить модель';
        }
    };

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