document.addEventListener('DOMContentLoaded', () => {
    // 1. Получение элементов
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const loading = document.getElementById('loading');
    const columnSelection = document.getElementById('columnSelection');
    const trainButton = document.getElementById('trainButton');
    const resultBoxes = document.querySelectorAll('.result-box');
    const errorBox = document.getElementById('error');

    const alphaGroup = document.getElementById('alphaGroup');
    const maxIterGroup = document.getElementById('maxIterGroup');
    const modelSelect = document.getElementById('paramPenalty');
    const alphaInput = document.getElementById('paramAlpha');
    const maxIterInput = document.getElementById('paramMaxIter');
    
    const API_URL = 'http://127.0.0.1:8000/linear_regression/';

    // === Глобальная переменная для хранения файла ===
    let selectedFile = null;

    let selectedColumns = {
        target: null,
        features: []
    };
    
    updateParamVisibility();

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
        resultBoxes.forEach(box => {box.classList.remove('show');})
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
        
        // === Валидация alpha ===
        const alphaInput = document.getElementById('paramAlpha');
        const alphaError = document.getElementById('alphaError');
        const alphaValue = parseFloat(alphaInput.value);
        
        if (isNaN(alphaValue) || alphaValue < 0.01 || alphaValue > 1000) {
            alphaError.style.display = 'block';
            alphaInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            alphaError.style.display = 'none';
            alphaInput.style.borderColor = '#ddd';
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
        resultBoxes.forEach(box => {box.classList.remove('show');})
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
        formData.append('alpha', document.getElementById('paramAlpha').value);
        formData.append('penalty', document.getElementById('paramPenalty').value);
        formData.append('max_iter', document.getElementById('paramMaxIter').value);
        formData.append('test_size', document.getElementById('paramTestSize').value);
        

        // Сохраняем выбор столбцов
        selectedColumns.target = document.getElementById('targetColumn').value;
        selectedColumns.features = Array.from(document.querySelectorAll('.feature-checkbox:checked')).map(cb => cb.value);
        formData.append('target_column', selectedColumns.target);
        formData.append('feature_columns', JSON.stringify(selectedColumns.features));

        formData.append('fit_intercept', document.getElementById('paramFitIntercept').checked);

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
        document.getElementById('r2Score').textContent = data.r2_score?.toFixed(4) || '-';
        document.getElementById('rmse').textContent = data.rmse?.toFixed(4) || '-';
        document.getElementById('mae').textContent = data.mae?.toFixed(4) || '-';
        document.getElementById('mse').textContent = data.mse?.toFixed(4) || '-';
        // Графики
        if (data.predicted_vs_actual) {
           drawPredictedVsActual(data.predicted_vs_actual);
        }
        if (data.residuals) {
           drawResiduals(data.residuals);
        }
        if (data.feature_importance && Object.keys(data.feature_importance).length > 0) {
           drawFeatureImportance(data.feature_importance);
        }
        if (data.convergence) {
            document.getElementById('nIter').textContent = data.convergence.n_iter || 'N/A';
            document.getElementById('converged').textContent = data.convergence.converged ? 'Да' : 'Нет';
        }
        resultBoxes.forEach(box => {box.classList.add('show');})
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

    // === График: Важность признаков ===
    function drawFeatureImportance(importance) {
        const sorted = Object.entries(importance)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 10);

        const features = sorted.map(([name, _]) => name);
        const coefficients = sorted.map(([_, value]) => value);
        const colors = coefficients.map(v => v >= 0 ? 'rgba(76, 175, 80, 0.7)' : 'rgba(244, 67, 54, 0.7)');

        const trace = {
            x: coefficients,
            y: features,
            type: 'bar',
            orientation: 'h',
            marker: { color: colors },
            text: coefficients.map(v => v.toFixed(3)),
            textposition: 'auto'
        };

        const layout = {
            title: 'Коэффициенты модели',
            xaxis: { title: 'Значение' },
            yaxis: { title: 'Признак' },
            margin: { t: 40, l: 150, r: 20, b: 50 }
        };

        Plotly.newPlot('featureImportancePlot', [trace], layout, { responsive: true });

        setTimeout(() => {
            Plotly.Plots.resize('featureImportancePlot');
        }, 100);

        document.getElementById('featureImportanceChart').style.display = 'block';
    }

    
    // Отображение ошибки
    function showError(message) {
        errorBox.textContent = '❌ ' + message;
        errorBox.classList.add('show');
    }

    function updateParamVisibility() {
        const type = modelSelect.value;
        if (type === 'none') {
            // OLS: аналитический метод
            alphaGroup.style.display = 'none';
            maxIterGroup.style.display = 'none';
            alphaInput.disabled = true;
            maxIterInput.disabled = true;

        } else if (type === 'l2') {
            // Ridge: по умолчанию аналитический, max_iter не нужен
            alphaGroup.style.display = 'block';
            maxIterGroup.style.display = 'none'; // Или 'block', если solver='sag'
            maxIterInput.disabled = true;

        } else if (type === 'l1') {
            // Lasso: итерационный, оба параметра нужны
            alphaGroup.style.display = 'block';
            maxIterGroup.style.display = 'block';
            alphaInput.disabled = false;
            maxIterInput.disabled = false;
        }
    }

    modelSelect.addEventListener('change', updateParamVisibility);
});