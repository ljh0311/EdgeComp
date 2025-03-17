/**
 * Baby Monitor System
 * Metrics Page JavaScript
 */

// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const currentFps = document.getElementById('currentFps');
const currentDetections = document.getElementById('currentDetections');
const currentCpu = document.getElementById('currentCpu');
const currentMemory = document.getElementById('currentMemory');
const systemUptime = document.getElementById('systemUptime');
const systemUptimeInfo = document.getElementById('systemUptimeInfo');
const personDetectorInfo = document.getElementById('personDetectorInfo');
const emotionDetectorInfo = document.getElementById('emotionDetectorInfo');
const cameraResolution = document.getElementById('cameraResolution');
const audioSampleRate = document.getElementById('audioSampleRate');
const detectionThreshold = document.getElementById('detectionThreshold');
const alertHistory = document.getElementById('alertHistory');
const exportAlertsBtn = document.getElementById('exportAlerts');
const clearDetectionLogBtn = document.getElementById('clearDetectionLog');
const detectionLogContainer = document.getElementById('detectionLogContainer');

// Detection type counters
const faceDetectionCount = document.getElementById('faceDetectionCount');
const upperBodyDetectionCount = document.getElementById('upperBodyDetectionCount');
const fullBodyDetectionCount = document.getElementById('fullBodyDetectionCount');
const lowerBodyDetectionCount = document.getElementById('lowerBodyDetectionCount');

// Emotion percentages
const cryingPercentage = document.getElementById('cryingPercentage');
const laughingPercentage = document.getElementById('laughingPercentage');
const babblingPercentage = document.getElementById('babblingPercentage');
const silencePercentage = document.getElementById('silencePercentage');

// Health bars
const cpuHealthBar = document.getElementById('cpuHealthBar');
const cpuHealthValue = document.getElementById('cpuHealthValue');
const memoryHealthBar = document.getElementById('memoryHealthBar');
const memoryHealthValue = document.getElementById('memoryHealthValue');
const diskHealthBar = document.getElementById('diskHealthBar');
const diskHealthValue = document.getElementById('diskHealthValue');

// Status indicators
const cameraStatus = document.getElementById('cameraStatus');
const audioStatus = document.getElementById('audioStatus');
const personDetectorStatus = document.getElementById('personDetectorStatus');
const emotionDetectorStatus = document.getElementById('emotionDetectorStatus');

// System info
const osInfo = document.getElementById('osInfo');
const pythonVersion = document.getElementById('pythonVersion');
const opencvVersion = document.getElementById('opencvVersion');

// Time range buttons
const timeRange1h = document.getElementById('timeRange1h');
const timeRange3h = document.getElementById('timeRange3h');
const timeRange24h = document.getElementById('timeRange24h');

// Charts
let fpsChart;
let detectionCountChart;
let emotionDistributionChart;
let detectionTypesChart;
let cpuUsageChart;
let memoryUsageChart;

// Chart data
const chartData = {
    fps: {
        labels: [],
        values: []
    },
    detectionCount: {
        labels: [],
        values: []
    },
    emotionDistribution: {
        labels: ['Crying', 'Laughing', 'Babbling', 'Silence'],
        values: [0, 0, 0, 0]
    },
    detectionTypes: {
        labels: ['Face', 'Upper Body', 'Full Body', 'Lower Body'],
        values: [0, 0, 0, 0]
    },
    cpuUsage: {
        labels: [],
        values: []
    },
    memoryUsage: {
        labels: [],
        values: []
    }
};

// Detection log
const detectionLog = [];
const MAX_LOG_ENTRIES = 20;

// System start time
let startTime = new Date();

// Current time range in hours
let currentTimeRange = 1;

// Initialize the metrics page
function initMetrics() {
    // Initialize charts
    initCharts();
    
    // Set up event listeners
    exportAlertsBtn.addEventListener('click', exportAlerts);
    timeRange1h.addEventListener('click', () => setTimeRange(1));
    timeRange3h.addEventListener('click', () => setTimeRange(3));
    timeRange24h.addEventListener('click', () => setTimeRange(24));
    
    if (clearDetectionLogBtn) {
        clearDetectionLogBtn.addEventListener('click', clearDetectionLog);
    }
    
    // Start uptime counter
    updateUptime();
    setInterval(updateUptime, 1000);
    
    // Set initial system info
    setSystemInfo();
    
    // Set up Socket.IO event handlers
    setupSocketEvents();
    
    // Add initial alert
    addAlert('Metrics dashboard initialized', 'info');
}

// Set up Socket.IO event handlers
function setupSocketEvents() {
    socket.on('connect', () => {
        console.log('Connected to server');
        updateComponentStatus('cameraStatus', true);
        updateComponentStatus('audioStatus', true);
        updateComponentStatus('personDetectorStatus', true);
        updateComponentStatus('emotionDetectorStatus', true);
        addAlert('Connected to server', 'info');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateComponentStatus('cameraStatus', false);
        updateComponentStatus('audioStatus', false);
        updateComponentStatus('personDetectorStatus', false);
        updateComponentStatus('emotionDetectorStatus', false);
        addAlert('Disconnected from server', 'warning');
    });
    
    socket.on('detection_update', (data) => {
        updateDetectionInfo(data);
    });
    
    socket.on('emotion_update', (data) => {
        updateEmotionInfo(data);
    });
    
    socket.on('metrics_update', (data) => {
        updateMetrics(data);
    });
    
    socket.on('alert', (data) => {
        addAlert(data.message, data.type);
    });
}

// Initialize all charts
function initCharts() {
    // Set Chart.js defaults for dark theme
    Chart.defaults.color = '#aaaaaa';
    Chart.defaults.borderColor = '#333333';
    
    // FPS Chart
    const fpsCtx = document.getElementById('fpsChart').getContext('2d');
    fpsChart = new Chart(fpsCtx, {
        type: 'line',
        data: {
            labels: chartData.fps.labels,
            datasets: [{
                label: 'FPS',
                data: chartData.fps.values,
                borderColor: '#0dcaf0',
                backgroundColor: 'rgba(13, 202, 240, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Frames Per Second')
    });
    
    // Detection Count Chart
    const detectionCountCtx = document.getElementById('detectionCountChart').getContext('2d');
    detectionCountChart = new Chart(detectionCountCtx, {
        type: 'line',
        data: {
            labels: chartData.detectionCount.labels,
            datasets: [{
                label: 'Detections',
                data: chartData.detectionCount.values,
                borderColor: '#198754',
                backgroundColor: 'rgba(25, 135, 84, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Detection Count')
    });
    
    // Emotion Distribution Chart
    const emotionDistributionCtx = document.getElementById('emotionDistributionChart').getContext('2d');
    emotionDistributionChart = new Chart(emotionDistributionCtx, {
        type: 'doughnut',
        data: {
            labels: chartData.emotionDistribution.labels,
            datasets: [{
                data: chartData.emotionDistribution.values,
                backgroundColor: [
                    '#dc3545', // Crying - Red
                    '#198754', // Laughing - Green
                    '#0dcaf0', // Babbling - Cyan
                    '#6c757d'  // Silence - Gray
                ],
                borderWidth: 1,
                borderColor: '#252525'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#e0e0e0',
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                            return `${label}: ${percentage}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Detection Types Chart
    const detectionTypesCtx = document.getElementById('detectionTypesChart').getContext('2d');
    detectionTypesChart = new Chart(detectionTypesCtx, {
        type: 'bar',
        data: {
            labels: chartData.detectionTypes.labels,
            datasets: [{
                label: 'Count',
                data: chartData.detectionTypes.values,
                backgroundColor: [
                    'rgba(0, 255, 0, 0.7)',   // Face - Green
                    'rgba(255, 0, 0, 0.7)',   // Upper Body - Red
                    'rgba(0, 0, 255, 0.7)',   // Full Body - Blue
                    'rgba(255, 255, 0, 0.7)'  // Lower Body - Yellow
                ],
                borderColor: [
                    'rgb(0, 255, 0)',
                    'rgb(255, 0, 0)',
                    'rgb(0, 0, 255)',
                    'rgb(255, 255, 0)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
    
    // CPU Usage Chart
    const cpuUsageCtx = document.getElementById('cpuUsageChart').getContext('2d');
    cpuUsageChart = new Chart(cpuUsageCtx, {
        type: 'line',
        data: {
            labels: chartData.cpuUsage.labels,
            datasets: [{
                label: 'CPU Usage (%)',
                data: chartData.cpuUsage.values,
                borderColor: '#fd7e14',
                backgroundColor: 'rgba(253, 126, 20, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('CPU Usage (%)', 0, 100)
    });
    
    // Memory Usage Chart
    const memoryUsageCtx = document.getElementById('memoryUsageChart').getContext('2d');
    memoryUsageChart = new Chart(memoryUsageCtx, {
        type: 'line',
        data: {
            labels: chartData.memoryUsage.labels,
            datasets: [{
                label: 'Memory Usage (%)',
                data: chartData.memoryUsage.values,
                borderColor: '#6f42c1',
                backgroundColor: 'rgba(111, 66, 193, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Memory Usage (%)', 0, 100)
    });
}

// Get common chart options
function getChartOptions(title, min = null, max = null) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                display: true,
                title: {
                    display: false
                },
                ticks: {
                    maxTicksLimit: 5,
                    color: '#aaaaaa'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)'
                }
            },
            y: {
                display: true,
                beginAtZero: true,
                suggestedMin: min,
                suggestedMax: max,
                ticks: {
                    color: '#aaaaaa'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)'
                }
            }
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                borderColor: '#555555',
                borderWidth: 1
            }
        }
    };
}

// Update detection information
function updateDetectionInfo(data) {
    if (!data) return;
    
    // Update current detection count
    if (currentDetections) {
        currentDetections.textContent = data.count || 0;
    }
    
    // Update detection types
    let faces = 0;
    let upperBodies = 0;
    let fullBodies = 0;
    let lowerBodies = 0;
    
    if (data.detections && Array.isArray(data.detections)) {
        data.detections.forEach(det => {
            if (det.class === 'face') faces++;
            if (det.class === 'upper_body') upperBodies++;
            if (det.class === 'full_body') fullBodies++;
            if (det.class === 'lower_body') lowerBodies++;
        });
    }
    
    // Update detection type counters
    if (faceDetectionCount) faceDetectionCount.textContent = faces;
    if (upperBodyDetectionCount) upperBodyDetectionCount.textContent = upperBodies;
    if (fullBodyDetectionCount) fullBodyDetectionCount.textContent = fullBodies;
    if (lowerBodyDetectionCount) lowerBodyDetectionCount.textContent = lowerBodies;
    
    // Update detection types chart
    if (detectionTypesChart) {
        detectionTypesChart.data.datasets[0].data = [faces, upperBodies, fullBodies, lowerBodies];
        detectionTypesChart.update();
    }
    
    // Update detection count chart
    addTimeSeriesData(chartData.detectionCount, data.count || 0, detectionCountChart);
    
    // Update FPS chart
    addTimeSeriesData(chartData.fps, data.fps || 0, fpsChart);
    
    // Update current FPS
    if (currentFps) {
        currentFps.textContent = Math.round(data.fps) || 0;
    }
    
    // Add to detection log if count > 0
    if (data.count > 0) {
        addToDetectionLog('person', data.count);
    }
    
    // Update component status
    updateComponentStatus('personDetectorStatus', true);
}

// Update emotion information
function updateEmotionInfo(data) {
    if (!data || !data.emotions) return;
    
    // Update emotion distribution chart
    const emotions = data.emotions;
    const emotionValues = [
        emotions.crying || 0,
        emotions.laughing || 0,
        emotions.babbling || 0,
        emotions.silence || 0
    ];
    
    if (emotionDistributionChart) {
        emotionDistributionChart.data.datasets[0].data = emotionValues;
        emotionDistributionChart.update();
    }
    
    // Calculate percentages
    const total = emotionValues.reduce((a, b) => a + b, 0);
    const percentages = emotionValues.map(value => total > 0 ? Math.round((value / total) * 100) : 0);
    
    // Update emotion percentage displays
    if (cryingPercentage) cryingPercentage.textContent = `${percentages[0]}%`;
    if (laughingPercentage) laughingPercentage.textContent = `${percentages[1]}%`;
    if (babblingPercentage) babblingPercentage.textContent = `${percentages[2]}%`;
    if (silencePercentage) silencePercentage.textContent = `${percentages[3]}%`;
    
    // Add to detection log if not silence and confidence is high enough
    if (data.emotion !== 'silence' && data.confidence > 0.3) {
        addToDetectionLog('emotion', data.emotion, Math.round(data.confidence * 100));
    }
    
    // Update component status
    updateComponentStatus('emotionDetectorStatus', true);
    updateComponentStatus('audioStatus', true);
}

// Update metrics
function updateMetrics(data) {
    if (!data || !data.current) return;
    
    const current = data.current;
    
    // Update CPU and memory usage
    const cpuUsage = current.cpu_usage || 0;
    const memoryUsage = current.memory_usage || 0;
    
    // Update current values
    if (currentCpu) {
        currentCpu.textContent = `${Math.round(cpuUsage)}%`;
    }
    
    if (currentMemory) {
        currentMemory.textContent = `${Math.round(memoryUsage)}%`;
    }
    
    // Update health bars
    updateHealthBar(cpuHealthBar, cpuHealthValue, cpuUsage);
    updateHealthBar(memoryHealthBar, memoryHealthValue, memoryUsage);
    
    // Simulate disk usage (random value between 40-50%)
    const diskUsage = 40 + Math.random() * 10;
    updateHealthBar(diskHealthBar, diskHealthValue, diskUsage);
    
    // Update charts
    addTimeSeriesData(chartData.cpuUsage, cpuUsage, cpuUsageChart);
    addTimeSeriesData(chartData.memoryUsage, memoryUsage, memoryUsageChart);
}

// Update health bar
function updateHealthBar(bar, valueElement, percentage) {
    if (!bar || !valueElement) return;
    
    // Set width
    bar.style.width = `${percentage}%`;
    
    // Set text
    valueElement.textContent = `${Math.round(percentage)}%`;
    
    // Set color based on value
    bar.className = 'health-bar';
    if (percentage < 50) {
        bar.classList.add('health-good');
    } else if (percentage < 80) {
        bar.classList.add('health-warning');
    } else {
        bar.classList.add('health-critical');
    }
}

// Update component status
function updateComponentStatus(elementId, isActive) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    if (isActive) {
        element.className = 'status-badge status-active';
        element.textContent = 'Active';
    } else {
        element.className = 'status-badge status-inactive';
        element.textContent = 'Inactive';
    }
}

// Add time series data
function addTimeSeriesData(dataSet, value, chart) {
    if (!chart) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // Add data
    dataSet.labels.push(timeString);
    dataSet.values.push(value);
    
    // Limit data points based on time range
    const maxPoints = currentTimeRange * 60; // 1 point per minute
    if (dataSet.labels.length > maxPoints) {
        dataSet.labels.shift();
        dataSet.values.shift();
    }
    
    // Update chart
    chart.update();
}

// Update uptime
function updateUptime() {
    if (!systemUptime && !systemUptimeInfo) return;
    
    const now = new Date();
    const diff = now - startTime;
    
    // Calculate hours, minutes, seconds
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    // Format as HH:MM:SS
    const uptimeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    // Update uptime displays
    if (systemUptime) systemUptime.textContent = uptimeString;
    if (systemUptimeInfo) systemUptimeInfo.textContent = uptimeString;
}

// Set system information
function setSystemInfo() {
    // Set detector info
    if (personDetectorInfo) personDetectorInfo.textContent = 'Haar Cascade (OpenCV)';
    if (emotionDetectorInfo) emotionDetectorInfo.textContent = 'Audio Analysis (Dummy Model)';
    
    // Set camera resolution
    if (cameraResolution) cameraResolution.textContent = '640x480';
    
    // Set audio sample rate
    if (audioSampleRate) audioSampleRate.textContent = '16000 Hz';
    
    // Set detection threshold
    if (detectionThreshold) detectionThreshold.textContent = '0.7';
    
    // Set OS info
    if (osInfo) {
        const userAgent = navigator.userAgent;
        if (userAgent.indexOf('Windows') !== -1) osInfo.textContent = 'Windows';
        else if (userAgent.indexOf('Mac') !== -1) osInfo.textContent = 'macOS';
        else if (userAgent.indexOf('Linux') !== -1) osInfo.textContent = 'Linux';
        else osInfo.textContent = 'Unknown';
    }
    
    // Set Python version (simulated)
    if (pythonVersion) pythonVersion.textContent = '3.8.10';
    
    // Set OpenCV version (simulated)
    if (opencvVersion) opencvVersion.textContent = '4.5.4';
}

// Add alert to history
function addAlert(message, type = 'info') {
    if (!alertHistory) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const alertItem = document.createElement('div');
    alertItem.className = 'alert-item';
    
    // Add color based on type
    if (type === 'crying' || type === 'alert' || type === 'error') {
        alertItem.classList.add('alert-danger');
    } else if (type === 'warning') {
        alertItem.classList.add('alert-warning');
    } else if (type === 'info') {
        alertItem.classList.add('alert-info');
    }
    
    const alertTime = document.createElement('div');
    alertTime.className = 'alert-time';
    alertTime.textContent = timeString;
    
    const alertMessage = document.createElement('div');
    alertMessage.className = 'alert-message';
    alertMessage.textContent = message;
    
    alertItem.appendChild(alertTime);
    alertItem.appendChild(alertMessage);
    
    // Add to container at the top
    alertHistory.insertBefore(alertItem, alertHistory.firstChild);
    
    // Limit to 20 alerts
    while (alertHistory.children.length > 20) {
        alertHistory.removeChild(alertHistory.lastChild);
    }
}

// Add to detection log
function addToDetectionLog(type, value, confidence = null) {
    if (!detectionLogContainer) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // Create log entry
    const logEntry = {
        time: timeString,
        type: type,
        value: value,
        confidence: confidence,
        timestamp: now.getTime()
    };
    
    // Add to log
    detectionLog.unshift(logEntry);
    
    // Limit log size
    if (detectionLog.length > MAX_LOG_ENTRIES) {
        detectionLog.pop();
    }
    
    // Update log display
    updateDetectionLogDisplay();
}

// Update detection log display
function updateDetectionLogDisplay() {
    if (!detectionLogContainer) return;
    
    // Clear container
    detectionLogContainer.innerHTML = '';
    
    // If no entries, show message
    if (detectionLog.length === 0) {
        const emptyMessage = document.createElement('div');
        emptyMessage.className = 'alert-item';
        emptyMessage.textContent = 'No detections yet';
        detectionLogContainer.appendChild(emptyMessage);
        return;
    }
    
    // Add entries
    detectionLog.forEach(entry => {
        const logItem = document.createElement('div');
        logItem.className = 'alert-item';
        
        const logTime = document.createElement('div');
        logTime.className = 'alert-time';
        logTime.textContent = entry.time;
        
        const logMessage = document.createElement('div');
        logMessage.className = 'alert-message';
        
        if (entry.type === 'person') {
            logMessage.innerHTML = `<i class="bi bi-people"></i> Detected ${entry.value} person(s)`;
            logItem.classList.add('alert-info');
        } else if (entry.type === 'emotion') {
            let icon = 'emoji-neutral';
            let alertClass = 'alert-info';
            
            if (entry.value === 'crying') {
                icon = 'emoji-frown';
                alertClass = 'alert-danger';
            } else if (entry.value === 'laughing') {
                icon = 'emoji-laughing';
                alertClass = 'alert-info';
            } else if (entry.value === 'babbling') {
                icon = 'chat-dots';
                alertClass = 'alert-info';
            }
            
            logMessage.innerHTML = `<i class="bi bi-${icon}"></i> Detected ${entry.value}`;
            if (entry.confidence !== null) {
                logMessage.innerHTML += ` (${entry.confidence}%)`;
            }
            
            logItem.classList.add(alertClass);
        }
        
        logItem.appendChild(logTime);
        logItem.appendChild(logMessage);
        
        detectionLogContainer.appendChild(logItem);
    });
}

// Clear detection log
function clearDetectionLog() {
    detectionLog.length = 0;
    updateDetectionLogDisplay();
}

// Export alerts
function exportAlerts() {
    if (!alertHistory || alertHistory.children.length === 0) {
        alert('No alerts to export');
        return;
    }
    
    let csvContent = 'data:text/csv;charset=utf-8,';
    csvContent += 'Time,Type,Message\n';
    
    // Get all alert items
    const alertItems = alertHistory.querySelectorAll('.alert-item');
    alertItems.forEach(item => {
        const time = item.querySelector('.alert-time').textContent;
        const message = item.querySelector('.alert-message').textContent;
        let type = 'info';
        
        if (item.classList.contains('alert-danger')) {
            type = 'danger';
        } else if (item.classList.contains('alert-warning')) {
            type = 'warning';
        }
        
        csvContent += `"${time}","${type}","${message}"\n`;
    });
    
    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `baby-monitor-alerts-${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    
    // Trigger download
    link.click();
    
    // Clean up
    document.body.removeChild(link);
}

// Set time range
function setTimeRange(hours) {
    currentTimeRange = hours;
    
    // Update active button
    timeRange1h.classList.remove('active');
    timeRange3h.classList.remove('active');
    timeRange24h.classList.remove('active');
    
    if (hours === 1) {
        timeRange1h.classList.add('active');
    } else if (hours === 3) {
        timeRange3h.classList.add('active');
    } else if (hours === 24) {
        timeRange24h.classList.add('active');
    }
    
    // Clear chart data
    chartData.fps.labels = [];
    chartData.fps.values = [];
    chartData.detectionCount.labels = [];
    chartData.detectionCount.values = [];
    chartData.cpuUsage.labels = [];
    chartData.cpuUsage.values = [];
    chartData.memoryUsage.labels = [];
    chartData.memoryUsage.values = [];
    
    // Update charts
    fpsChart.update();
    detectionCountChart.update();
    cpuUsageChart.update();
    memoryUsageChart.update();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initMetrics); 