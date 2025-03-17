/**
 * Baby Monitor System
 * Main Dashboard JavaScript
 */

// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const videoFeed = document.getElementById('videoFeed');
const noVideoOverlay = document.getElementById('noVideoOverlay');
const detectionOverlay = document.getElementById('detectionOverlay');
const overlayPersonCount = document.getElementById('overlayPersonCount');
const overlayEmotion = document.getElementById('overlayEmotion');
const overlayEmotionConfidence = document.getElementById('overlayEmotionConfidence');
const overlayFps = document.getElementById('overlayFps');
const toggleCameraBtn = document.getElementById('toggleCamera');
const cameraButtonText = document.getElementById('cameraButtonText');
const cameraStatus = document.getElementById('cameraStatus');
const audioStatus = document.getElementById('audioStatus');
const personDetectorStatus = document.getElementById('personDetectorStatus');
const emotionDetectorStatus = document.getElementById('emotionDetectorStatus');
const systemFps = document.getElementById('systemFps');
const alertsContainer = document.getElementById('alertsContainer');
const clearAlertsBtn = document.getElementById('clearAlerts');
const alertSound = document.getElementById('alertSound');
const detectionLogTable = document.getElementById('detectionLogTable');
const clearLogBtn = document.getElementById('clearLog');

// Person detection elements
const personCount = document.getElementById('personCount');
const personFps = document.getElementById('personFps');
const faceCount = document.getElementById('faceCount');
const upperBodyCount = document.getElementById('upperBodyCount');
const fullBodyCount = document.getElementById('fullBodyCount');

// Emotion detection elements
const emotionIcon = document.getElementById('emotionIcon');
const currentEmotion = document.getElementById('currentEmotion');
const emotionConfidence = document.getElementById('emotionConfidence');
const cryingProgress = document.getElementById('cryingProgress');
const laughingProgress = document.getElementById('laughingProgress');
const babblingProgress = document.getElementById('babblingProgress');
const silenceProgress = document.getElementById('silenceProgress');
const cryingValue = document.getElementById('cryingValue');
const laughingValue = document.getElementById('laughingValue');
const babblingValue = document.getElementById('babblingValue');
const silenceValue = document.getElementById('silenceValue');

// Performance chart
let performanceChart;
const performanceData = {
    labels: [],
    fps: [],
    detections: []
};

// Detection log
const detectionLog = [];
const MAX_LOG_ENTRIES = 50;

// System state
let lastDetectionCount = 0;
let lastEmotionUpdate = Date.now();
let emotionUpdateInterval = 2000; // 2 seconds
let currentEmotionState = 'unknown';
let currentEmotionConfidence = 0;

// Initialize the dashboard
function initDashboard() {
    // Initialize performance chart
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: performanceData.labels,
            datasets: [
                {
                    label: 'FPS',
                    data: performanceData.fps,
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Detections',
                    data: performanceData.detections,
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 5
                    }
                },
                y: {
                    display: true,
                    beginAtZero: true,
                    suggestedMax: 30
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });

    // Set up event listeners
    toggleCameraBtn.addEventListener('click', toggleCamera);
    clearAlertsBtn.addEventListener('click', clearAlerts);
    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', clearDetectionLog);
    }

    // Check camera status on load
    updateCameraStatus(true);
    
    // Initialize detection log
    updateDetectionLogTable();
    
    // Add initial system alert
    addAlert('Baby monitor system is now running', 'info');
    
    // Set up periodic checks for emotion updates
    setInterval(checkEmotionTimeout, 5000);
    
    // Update overlay periodically
    setInterval(updateOverlay, 1000);
}

// Socket.IO event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    addAlert('System connected');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateStatus('cameraStatus', false);
    updateStatus('audioStatus', false);
    updateStatus('personDetectorStatus', false);
    updateStatus('emotionDetectorStatus', false);
    addAlert('System disconnected');
});

socket.on('camera_status', (data) => {
    updateCameraStatus(data.enabled);
});

socket.on('detection_update', (data) => {
    updateDetectionInfo(data);
    
    // Add person detection to log if count changed
    if (data.count !== lastDetectionCount) {
        if (data.count > 0) {
            addToDetectionLog('person', data.count, Math.round(data.fps), 'normal');
        }
        lastDetectionCount = data.count;
    }
});

socket.on('emotion_update', (data) => {
    updateEmotionInfo(data);
    lastEmotionUpdate = Date.now();
    currentEmotionState = data.emotion;
    currentEmotionConfidence = data.confidence;
    
    // Add emotion detection to log if not silence
    if (data.emotion !== 'silence' && data.confidence > 0.3) {
        const status = data.emotion === 'crying' && data.confidence > 0.7 ? 'alert' : 
                      (data.emotion === 'crying' ? 'warning' : 'normal');
        addToDetectionLog('emotion', data.emotion, Math.round(data.confidence * 100), status);
    }
});

socket.on('metrics_update', (data) => {
    updateMetrics(data);
});

socket.on('alert', (data) => {
    handleAlert(data);
});

// Function to toggle camera
function toggleCamera() {
    socket.emit('toggle_camera');
}

// Update camera status
function updateCameraStatus(enabled) {
    if (enabled) {
        videoFeed.classList.remove('d-none');
        noVideoOverlay.classList.add('d-none');
        detectionOverlay.classList.remove('d-none');
        cameraButtonText.textContent = 'Stop Camera';
        updateStatus('cameraStatus', true);
    } else {
        videoFeed.classList.add('d-none');
        noVideoOverlay.classList.remove('d-none');
        detectionOverlay.classList.add('d-none');
        cameraButtonText.textContent = 'Start Camera';
        updateStatus('cameraStatus', false);
    }
}

// Update status indicators
function updateStatus(elementId, isActive) {
    const element = document.getElementById(elementId);
    if (element) {
        if (isActive) {
            element.className = 'badge bg-success';
            element.textContent = 'Active';
        } else {
            element.className = 'badge bg-danger';
            element.textContent = 'Inactive';
        }
    }
}

// Update detection information
function updateDetectionInfo(data) {
    // Update person count
    if (personCount) {
        personCount.textContent = data.count || 0;
    }
    
    // Update overlay
    if (overlayPersonCount) {
        overlayPersonCount.textContent = data.count || 0;
    }
    
    // Update FPS
    if (personFps) {
        personFps.textContent = Math.round(data.fps) || 0;
    }
    
    if (overlayFps) {
        overlayFps.textContent = Math.round(data.fps) || 0;
    }
    
    // Update detection types
    let faces = 0;
    let upperBodies = 0;
    let fullBodies = 0;
    
    if (data.detections && Array.isArray(data.detections)) {
        data.detections.forEach(det => {
            if (det.class === 'face') faces++;
            if (det.class === 'upper_body') upperBodies++;
            if (det.class === 'full_body') fullBodies++;
        });
    }
    
    if (faceCount) faceCount.textContent = faces;
    if (upperBodyCount) upperBodyCount.textContent = upperBodies;
    if (fullBodyCount) fullBodyCount.textContent = fullBodies;
    
    // Update detector status
    updateStatus('personDetectorStatus', true);
}

// Update overlay with current detection information
function updateOverlay() {
    if (!detectionOverlay) return;
    
    // Update emotion in overlay
    if (overlayEmotion && currentEmotionState) {
        if (currentEmotionState === 'buffering' || currentEmotionState === 'unknown') {
            overlayEmotion.textContent = 'Unknown';
        } else {
            overlayEmotion.textContent = currentEmotionState.charAt(0).toUpperCase() + currentEmotionState.slice(1);
        }
    }
    
    if (overlayEmotionConfidence) {
        overlayEmotionConfidence.textContent = `${Math.round(currentEmotionConfidence * 100)}%`;
    }
    
    // Add alert class if crying detected
    if (currentEmotionState === 'crying' && currentEmotionConfidence > 0.5) {
        detectionOverlay.classList.add('bg-danger');
        detectionOverlay.classList.remove('bg-dark');
    } else {
        detectionOverlay.classList.remove('bg-danger');
        detectionOverlay.classList.add('bg-dark');
    }
}

// Check if emotion updates have timed out
function checkEmotionTimeout() {
    const now = Date.now();
    if (now - lastEmotionUpdate > emotionUpdateInterval) {
        // If no updates for a while, set to unknown
        if (currentEmotion) {
            currentEmotion.textContent = 'Unknown';
            currentEmotion.className = 'badge bg-secondary';
        }
        
        if (emotionIcon) {
            updateEmotionIcon('emoji-neutral', 'emotion-silence');
        }
        
        // Update overlay
        if (overlayEmotion) {
            overlayEmotion.textContent = 'Unknown';
        }
        
        if (overlayEmotionConfidence) {
            overlayEmotionConfidence.textContent = '0%';
        }
        
        // Update status
        updateStatus('emotionDetectorStatus', true);
        updateStatus('audioStatus', true);
        
        // Update state
        currentEmotionState = 'unknown';
        currentEmotionConfidence = 0;
    }
}

// Update emotion information
function updateEmotionInfo(data) {
    if (!data) return;
    
    // Update current emotion
    if (currentEmotion) {
        currentEmotion.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
        
        // Change badge color based on emotion
        if (data.emotion === 'crying') {
            currentEmotion.className = 'badge bg-danger';
            updateEmotionIcon('emoji-frown-fill', 'emotion-crying');
        } else if (data.emotion === 'laughing') {
            currentEmotion.className = 'badge bg-success';
            updateEmotionIcon('emoji-laughing-fill', 'emotion-laughing');
        } else if (data.emotion === 'babbling') {
            currentEmotion.className = 'badge bg-info';
            updateEmotionIcon('chat-dots-fill', 'emotion-babbling');
        } else {
            currentEmotion.className = 'badge bg-secondary';
            updateEmotionIcon('emoji-neutral-fill', 'emotion-silence');
        }
    }
    
    // Update confidence
    if (emotionConfidence) {
        const confidencePercent = Math.round(data.confidence * 100);
        emotionConfidence.textContent = `${confidencePercent}%`;
    }
    
    // Update progress bars
    if (data.emotions) {
        updateProgressBar(cryingProgress, data.emotions.crying || 0, cryingValue);
        updateProgressBar(laughingProgress, data.emotions.laughing || 0, laughingValue);
        updateProgressBar(babblingProgress, data.emotions.babbling || 0, babblingValue);
        updateProgressBar(silenceProgress, data.emotions.silence || 0, silenceValue);
    }
    
    // Update detector status
    updateStatus('emotionDetectorStatus', true);
    updateStatus('audioStatus', true);
}

// Update emotion icon
function updateEmotionIcon(iconName, emotionClass) {
    if (!emotionIcon) return;
    
    // Remove all emotion classes
    emotionIcon.classList.remove('emotion-crying', 'emotion-laughing', 'emotion-babbling', 'emotion-silence');
    
    // Add the current emotion class
    emotionIcon.classList.add(emotionClass);
    
    // Update the icon
    emotionIcon.innerHTML = `<i class="bi bi-${iconName} fs-1"></i>`;
}

// Update progress bar
function updateProgressBar(element, value, valueElement) {
    if (!element) return;
    const percent = Math.round(value * 100);
    element.style.width = `${percent}%`;
    element.setAttribute('aria-valuenow', percent);
    
    // Update value text if element exists
    if (valueElement) {
        valueElement.textContent = `${percent}%`;
    }
}

// Update metrics
function updateMetrics(data) {
    if (!data || !data.current) return;
    
    const current = data.current;
    
    // Update FPS
    if (systemFps) {
        systemFps.textContent = Math.round(current.fps) || 0;
    }
    
    // Update performance chart
    updatePerformanceChart(current.fps, current.detection_count);
}

// Update performance chart
function updatePerformanceChart(fps, detections) {
    // Add timestamp
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // Add data
    performanceData.labels.push(timeString);
    performanceData.fps.push(fps || 0);
    performanceData.detections.push(detections || 0);
    
    // Limit data points to last 20
    if (performanceData.labels.length > 20) {
        performanceData.labels.shift();
        performanceData.fps.shift();
        performanceData.detections.shift();
    }
    
    // Update chart
    if (performanceChart) {
        performanceChart.update();
    }
}

// Handle alerts
function handleAlert(data) {
    addAlert(data.message, data.type);
    
    // Play sound for crying alerts
    if (data.type === 'crying' && alertSound) {
        alertSound.play().catch(err => console.error('Error playing alert sound:', err));
    }
}

// Add alert to container
function addAlert(message, type = 'info') {
    if (!alertsContainer) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    const alertItem = document.createElement('div');
    alertItem.className = 'alert-item alert-highlight';
    
    const alertTime = document.createElement('div');
    alertTime.className = 'alert-time';
    alertTime.textContent = timeString;
    
    const alertMessage = document.createElement('div');
    alertMessage.className = 'alert-message';
    alertMessage.textContent = message;
    
    // Add color based on type
    if (type === 'crying' || type === 'alert') {
        alertItem.classList.add('alert-danger');
    } else if (type === 'warning') {
        alertItem.classList.add('alert-warning');
    } else if (type === 'info') {
        alertItem.classList.add('alert-info');
    }
    
    alertItem.appendChild(alertTime);
    alertItem.appendChild(alertMessage);
    
    // Add to container at the top
    alertsContainer.insertBefore(alertItem, alertsContainer.firstChild);
    
    // Limit to 20 alerts
    while (alertsContainer.children.length > 20) {
        alertsContainer.removeChild(alertsContainer.lastChild);
    }
}

// Clear all alerts
function clearAlerts() {
    if (alertsContainer) {
        alertsContainer.innerHTML = '';
    }
}

// Add entry to detection log
function addToDetectionLog(type, value, confidence, status) {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    // Create log entry
    const logEntry = {
        time: timeString,
        type: type,
        value: value,
        confidence: confidence,
        status: status,
        timestamp: now.getTime()
    };
    
    // Add to log
    detectionLog.unshift(logEntry);
    
    // Limit log size
    if (detectionLog.length > MAX_LOG_ENTRIES) {
        detectionLog.pop();
    }
    
    // Update table
    updateDetectionLogTable();
}

// Update detection log table
function updateDetectionLogTable() {
    if (!detectionLogTable) return;
    
    // Clear table
    detectionLogTable.innerHTML = '';
    
    // If no entries, show message
    if (detectionLog.length === 0) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 4;
        cell.className = 'text-center';
        cell.textContent = 'No detections yet';
        row.appendChild(cell);
        detectionLogTable.appendChild(row);
        return;
    }
    
    // Add entries
    detectionLog.forEach(entry => {
        const row = document.createElement('tr');
        
        // Time
        const timeCell = document.createElement('td');
        timeCell.textContent = entry.time;
        row.appendChild(timeCell);
        
        // Type and value
        const typeCell = document.createElement('td');
        if (entry.type === 'person') {
            typeCell.innerHTML = `<i class="bi bi-people"></i> ${entry.value} person(s)`;
        } else if (entry.type === 'emotion') {
            let icon = 'emoji-neutral';
            if (entry.value === 'crying') icon = 'emoji-frown';
            else if (entry.value === 'laughing') icon = 'emoji-laughing';
            else if (entry.value === 'babbling') icon = 'chat-dots';
            
            typeCell.innerHTML = `<i class="bi bi-${icon}"></i> ${entry.value.charAt(0).toUpperCase() + entry.value.slice(1)}`;
        }
        row.appendChild(typeCell);
        
        // Confidence
        const confidenceCell = document.createElement('td');
        confidenceCell.textContent = `${entry.confidence}${entry.type === 'emotion' ? '%' : ''}`;
        row.appendChild(confidenceCell);
        
        // Status
        const statusCell = document.createElement('td');
        let statusClass = 'bg-secondary';
        let statusText = 'Normal';
        
        if (entry.status === 'alert') {
            statusClass = 'bg-danger';
            statusText = 'Alert';
        } else if (entry.status === 'warning') {
            statusClass = 'bg-warning text-dark';
            statusText = 'Warning';
        } else if (entry.status === 'normal') {
            statusClass = 'bg-success';
            statusText = 'Normal';
        }
        
        statusCell.innerHTML = `<span class="badge ${statusClass}">${statusText}</span>`;
        row.appendChild(statusCell);
        
        detectionLogTable.appendChild(row);
    });
}

// Clear detection log
function clearDetectionLog() {
    detectionLog.length = 0;
    updateDetectionLogTable();
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', initDashboard); 