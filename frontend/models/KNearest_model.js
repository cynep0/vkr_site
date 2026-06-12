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
    const resultBoxClassification = document.getElementById('resultBoxClassification');
    const resultBoxRegression = document.getElementById('resultBoxRegression');
    const errorBox = document.getElementById('error');
    
    const API_URL = 'http://127.0.0.1:8000/k_nearest/';

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
        resultBoxRegression.classList.remove('show');
        resultBoxClassification.classList.remove('show');
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

            // Заполняем выбор целевой переменной
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

    // === Обновление параметра p ===
    window.updatePParam = function() {
        const metric = document.getElementById('paramMetric').value;
        const pGroup = document.getElementById('pParamGroup');
        const pSelect = document.getElementById('paramP');

        if (metric === 'euclidean') {
            pSelect.value = '2';
            pGroup.style.opacity = '0.5';
            pGroup.style.pointerEvents = 'none';
            document.getElementById('paramPValue').textContent = 2
        } else if (metric === 'manhattan') {
            pSelect.value = '1';
            pGroup.style.opacity = '0.5';
            pGroup.style.pointerEvents = 'none';
            document.getElementById('paramPValue').textContent = 1
        } else {
            pGroup.style.opacity = '1';
            pGroup.style.pointerEvents = 'auto';
        }
    };

    function validateParams() {
        let isValid = true;
        
        // === Валидация nEstimators ===
        const paramNNeighborsInput = document.getElementById('paramNNeighbors');
        const paramNNeighborsError = document.getElementById('paramNNeighborsError');
        const paramNNeighborsValue = parseInt(paramNNeighborsInput.value);

        if (isNaN(paramNNeighborsValue) || paramNNeighborsValue < 1 || paramNNeighborsValue > 50) {
            paramNNeighborsError.style.display = 'block';
            paramNNeighborsInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            paramNNeighborsError.style.display = 'none';
            paramNNeighborsInput.style.borderColor = '#ddd';
        }

        return isValid;
    }

    // Обучение модели (по кнопке)
    window.trainModel = async function() {
        // Сброс интерфейса
        resultBoxRegression.classList.remove('show');
        resultBoxClassification.classList.remove('show');
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
        formData.append('task_type', document.querySelector('input[name="taskType"]:checked').value);
        formData.append('n_neighbors', document.getElementById('paramNNeighbors').value);
        formData.append('weights', document.getElementById('paramWeights').value);
        formData.append('metric', document.getElementById('paramMetric').value);
        formData.append('p', document.getElementById('paramP').value);
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
        document.getElementById('trainSize').textContent = data.train_size;
        document.getElementById('testSize').textContent = data.test_size;
        document.getElementById('features').textContent = data.features.join(', ');
        document.getElementById('message').textContent = data.message;
        
        resultBox.classList.add('show');
        // Переключение метрик
        if (data.task_type === 'classification') {
            document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('rocAuc').textContent = data.roc_auc !== null ? data.roc_auc.toFixed(4) : 'N/A';
            document.getElementById('precision').textContent = data.precision?.toFixed(4) || '-';
            document.getElementById('recall').textContent = data.recall?.toFixed(4) || '-';
            document.getElementById('f1Score').textContent = data.f1_score?.toFixed(4) || '-';
            // Матрица ошибок
            if (data.confusion_matrix) {
                drawConfusionMatrix(data.confusion_matrix, data.confusion_matrix_labels);
            }
            resultBoxRegression.classList.remove('show');
            resultBoxClassification.classList.add('show');
        } else {
            document.getElementById('r2Score').textContent = data.r2_score?.toFixed(4) || '-';
            document.getElementById('rmse').textContent = data.rmse?.toFixed(4) || '-';
            document.getElementById('mae').textContent = data.mae?.toFixed(4) || '-';
            document.getElementById('mse').textContent = data.mse?.toFixed(4) || '-';
            // График Predicted vs Actual
            if (data.predicted_vs_actual) {
               drawPredictedVsActual(data.predicted_vs_actual);
            }
            //График Residuals
            if (data.residuals) {
                drawResiduals(data.residuals)
            }        
            resultBoxClassification.classList.remove('show');
            resultBoxRegression.classList.add('show');
        }
    }

    // === Функция рисования матрицы ошибок (Plotly) ===
    function drawConfusionMatrix(matrix, labels) {
        // Если метки не переданы, используем [0, 1]
        const classLabels = labels || [0, 1];

        // Создаем текст для ячеек
        const zText = matrix.map(row => row.map(val => val.toString()));

        const trace = {
            x: classLabels,
            y: classLabels,
            z: matrix,
            text: zText,
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true,
            texttemplate: '%{text}',
            textfont: {size: 14, color: matrix.some(row => row.some(v => v > 10)) ? 'white' : 'black'}
        };

        const layout = {
            xaxis: {title: 'Предсказанный класс'},
            yaxis: {title: 'Реальный класс', autorange: 'reversed'},
            margin: {t: 50, l: 50, r: 20, b: 50}
        };

        Plotly.newPlot('confusionMatrixPlot', [trace], layout, {responsive: true});

        setTimeout(() => {
            Plotly.Plots.resize('confusionMatrixPlot');
        }, 100);

        document.getElementById('confusionMatrixChart').style.display = 'block';
    }

    // === График: Predicted vs Actual ===
    function drawPredictedVsActual(data) {
        const actual = data.map(d => d.actual);
        const predicted = data.map(d => d.predicted);
        const minVal = Math.min(...actual, ...predicted);
        const maxVal = Math.max(...actual, ...predicted);

        const trace = {
            x: actual,
            y: predicted,
            mode: 'markers',
            type: 'scatter',
            marker: { color: 'rgba(102, 126, 234, 0.7)', size: 8 },
            name: 'Предсказания'
        };

        const diagonal = {
            x: [minVal, maxVal],
            y: [minVal, maxVal],
            mode: 'lines',
            type: 'scatter',
            line: { color: '#ff6b6b', width: 2, dash: 'dash' },
            name: 'Идеальное предсказание'
        };

        const layout = {
            title: 'Предсказание vs Реальность',
            xaxis: { title: 'Реальное значение' },
            yaxis: { title: 'Предсказанное значение' },
            margin: { t: 40, l: 50, r: 20, b: 50 }
        };

        Plotly.newPlot('predictedVsActualPlot', [diagonal, trace], layout, { responsive: true });

        setTimeout(() => {
            Plotly.Plots.resize('predictedVsActualPlot');
        }, 100);

        document.getElementById('predictedVsActualChart').style.display = 'block';
    }

    // === График: Остатки ===
    function drawResiduals(residuals) {
        const trace = {
            y: residuals,
            mode: 'markers',
            type: 'scatter',
            marker: { 
                color: residuals.map(r => r >= 0 ? 'rgba(76, 175, 80, 0.7)' : 'rgba(244, 67, 54, 0.7)'),
                size: 6 
            },
            name: 'Остатки'
        };

        const zeroLine = {
            x: [0, residuals.length],
            y: [0, 0],
            mode: 'lines',
            type: 'scatter',
            line: { color: '#333', width: 1, dash: 'dot' },
            name: 'Ноль'
        };

        const layout = {
            title: 'Остатки модели',
            xaxis: { title: 'Индекс' },
            yaxis: { title: 'Остаток' },
            margin: { t: 40, l: 50, r: 20, b: 50 }
        };

        Plotly.newPlot('residualsPlot', [zeroLine, trace], layout, { responsive: true });

        setTimeout(() => {
            Plotly.Plots.resize('residualsPlot');
        }, 100);

        document.getElementById('residualsChart').style.display = 'block';
    }

    // Отображение ошибки
    function showError(message) {
        errorBox.textContent = '❌ ' + message;
        errorBox.classList.add('show');
    }
});