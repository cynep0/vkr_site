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
    const resultBoxMetrics= document.getElementById('resultBoxMetrics');
    const errorBox = document.getElementById('error');
    
    const API_URL = 'http://127.0.0.1:8000/kmeans/';

    // === Глобальная переменная для хранения файла ===
    let selectedFile = null;

    let selectedColumns = {
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
        resultBoxMetrics.classList.remove('show');
        resultBox.classList.remove('show');
        errorBox.classList.remove('show');
        
        // Зона загрузки становится менее заметной
        dropZone.style.opacity = '0.6';
        dropZone.style.pointerEvents = 'none';
    }

    async function loadColumnNames(file) {
        const featuresList = document.getElementById('featuresList');
    
        // Читаем первую строку CSV для получения имён столбцов
        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            const firstLine = text.split('\n')[0];
            const columns = firstLine.split(',').map(col => col.trim().replace(/"/g, ''));

            // Заполняем выбор целевой переменной
            columns.forEach((col, index) => {
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

            // Добавляем обработчики на чекбоксы
            document.querySelectorAll('.feature-checkbox').forEach(cb => {
                cb.addEventListener('change', validateColumnSelection);
            });
        };
        reader.readAsText(file);
    }

    function validateColumnSelection() {
        const featureCheckboxes = document.querySelectorAll('.feature-checkbox:checked');
        const features = Array.from(featureCheckboxes).map(cb => cb.value);
        const warningDiv = document.getElementById('columnWarning');
        const warningText = document.getElementById('warningText');
        
        let warnings = [];
        // Проверка: выбраны ли признаки
        if (features.length < 2) {
            warnings.push('Выберите минимум 2 признака');
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
        
        // === Валидация nClusters ===
        const nClustersInput = document.getElementById('paramNClusters');
        const nClustersError = document.getElementById('nClustersError');
        const nClustersValue = parseInt(nClustersInput.value);

        if (isNaN(nClustersValue) || nClustersValue < 2 || nClustersValue > 20) {
            nClustersError.style.display = 'block';
            nClustersInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            nClustersError.style.display = 'none';
            nClustersInput.style.borderColor = '#ddd';
        }

        // === Валидация maxIter ===
        const maxIterInput = document.getElementById('paramMaxIter');
        const maxIterError = document.getElementById('maxIterError');
        const maxIterValue = parseInt(maxIterInput.value);
        
        if (isNaN(maxIterValue) || maxIterValue < 100 || maxIterValue > 1000) {
            maxIterError.style.display = 'block';
            maxIterInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            maxIterError.style.display = 'none';
            maxIterInput.style.borderColor = '#ddd';
        }

        // === Валидация nInits ===
        const nInitInput = document.getElementById('paramNInit');
        const nInitError = document.getElementById('nInitError');
        const nInitValue = parseInt(nInitInput.value);
        if (isNaN(nInitValue) || nInitValue < 1 || nInitValue > 50) {
            nInitError.style.display = 'block';
            nInitInput.style.borderColor = '#cc0000';
            isValid = false;
        } else {
            nInitError.style.display = 'none';
            nInitInput.style.borderColor = '#ddd';
        }

        return isValid;
    }

    // Обучение модели (по кнопке)
    window.trainModel = async function() {
        // Сброс интерфейса
        resultBoxMetrics.classList.remove('show');
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
        formData.append('feature_columns', JSON.stringify(selectedColumns.features));
        formData.append('n_clusters', document.getElementById('paramNClusters').value);
        formData.append('init', document.getElementById('paramInit').value);
        formData.append('max_iter', document.getElementById('paramMaxIter').value);
        formData.append('n_init', document.getElementById('paramNInit').value);

        // Сохраняем выбор столбцов
        selectedColumns.features = Array.from(document.querySelectorAll('.feature-checkbox:checked')).map(cb => cb.value);
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
        document.getElementById('features').textContent = data.features.join(', ');
        document.getElementById('message').textContent = data.message;
        document.getElementById('resultNClusters').textContent = data.n_clusters;
        document.getElementById('resultNSamples').textContent = data.n_samples;
        document.getElementById('pcaVariance').textContent = data.pca_explained_variance || '-';
        // Метрики
        document.getElementById('inertia').textContent = data.inertia?.toFixed(2) || '-';
        document.getElementById('silhouette').textContent = data.silhouette_score?.toFixed(4) || '-';
        document.getElementById('daviesBouldin').textContent = data.davies_bouldin_score?.toFixed(4) || '-';
        
        // Распределение по кластерам
        const distContainer = document.getElementById('clusterDistribution');
        distContainer.innerHTML = '';
        Object.entries(data.cluster_distribution || {}).forEach(([cluster, count]) => {
            // Создаём обёртку в едином стиле платформы
            const item = document.createElement('div');
            item.className = 'result-item';
                
            // Метка (название кластера)
            const label = document.createElement('span');
            label.className = 'result-label';
            label.textContent = `${cluster}:`;
                
            // Значение (количество объектов)
            const value = document.createElement('span');
            value.className = 'result-value';
            value.textContent = `${count} объектов`;
                
            item.appendChild(label);
            item.appendChild(value);
            distContainer.appendChild(item);
        });

        // График кластеров
        if (data.scatter_data && data.scatter_data.length > 0) {
            drawScatterPlot(data.scatter_data, data.centroid_data);
        }

        resultBox.classList.add('show');
        resultBoxMetrics.classList.add('show');
        
    }

    // === График: Scatter plot кластеров ===
    function drawScatterPlot(scatterData, centroidData) {
        // Группируем точки по кластерам
        const clusters = {};
        scatterData.forEach(point => {
            if (!clusters[point.cluster]) clusters[point.cluster] = { x: [], y: [] };
            clusters[point.cluster].x.push(point.x);
            clusters[point.cluster].y.push(point.y);
        });

        const traces = [];
        const colors = ['#667eea', '#28a745', '#ffc107', '#f44336', '#9c27b0', '#00bcd4', '#ff5722'];

        Object.entries(clusters).forEach(([cluster, points], i) => {
            traces.push({
                x: points.x,
                y: points.y,
                mode: 'markers',
                type: 'scatter',
                name: `Cluster ${cluster}`,
                marker: { 
                    color: colors[i % colors.length], 
                    size: 8,
                    opacity: 0.7
                }
            });
        });

        // Центроиды
        if (centroidData && centroidData.length > 0) {
            traces.push({
                x: centroidData.map(c => c.x),
                y: centroidData.map(c => c.y),
                mode: 'markers',
                type: 'scatter',
                name: 'Центроиды',
                marker: { 
                    color: '#000000', 
                    size: 15,
                    symbol: 'x',
                    line: { width: 3, color: '#ffffff' }
                }
            });
        }

        const layout = {
            title: 'Кластеры на плоскости (PCA)',
            xaxis: { title: 'PC1' },
            yaxis: { title: 'PC2' },
            legend: { x: 1, xanchor: 'right' },
            margin: { t: 50, l: 50, r: 20, b: 50 }
        };

        Plotly.newPlot('scatterPlot', traces, layout, { responsive: true });

        setTimeout(() => {
            Plotly.Plots.resize('scatterPlot');
        }, 100);

        document.getElementById('scatterChart').style.display = 'block';
    }

    // Отображение ошибки
    function showError(message) {
        errorBox.textContent = '❌ ' + message;
        errorBox.classList.add('show');
    }
});