// Metrics Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log("Metrics page initialized");
    
    // Socket connection status display
    const connectionStatus = document.createElement('div');
    connectionStatus.style.position = 'fixed';
    connectionStatus.style.bottom = '10px';
    connectionStatus.style.right = '10px';
    connectionStatus.style.padding = '5px 10px';
    connectionStatus.style.borderRadius = '5px';
    connectionStatus.style.fontSize = '12px';
    connectionStatus.style.zIndex = '1000';
    connectionStatus.style.backgroundColor = '#dc3545'; // Red initially
    connectionStatus.style.color = 'white';
    connectionStatus.innerHTML = 'Socket: Disconnected';
    document.body.appendChild(connectionStatus);

    // Add a refresh button right next to the connection status
    const refreshButton = document.createElement('button');
    refreshButton.style.position = 'fixed';
    refreshButton.style.bottom = '10px';
    refreshButton.style.right = '150px';
    refreshButton.style.padding = '5px 10px';
    refreshButton.style.borderRadius = '5px';
    refreshButton.style.fontSize = '12px';
    refreshButton.style.zIndex = '1000';
    refreshButton.style.backgroundColor = '#0dcaf0';
    refreshButton.style.color = 'white';
    refreshButton.style.border = 'none';
    refreshButton.innerHTML = 'Refresh Connection';
    document.body.appendChild(refreshButton);
    
    refreshButton.addEventListener('click', function() {
        console.log("Manually refreshing connection");
        addDetectionEvent(new Date().toLocaleString(), "Refreshing connection...");
        
        // Reconnect socket if needed
        if (!socketConnected) {
            socket.connect();
        }
        
        // Request fresh metrics data
        socket.emit('request_metrics', { timeRange: '1h' });
    });

    // Initialize charts
    const fpsChart = createLineChart('fpsChart', 'FPS', 'rgba(13, 202, 240, 0.5)', 'rgba(13, 202, 240, 1)');
    const detectionCountChart = createLineChart('detectionCountChart', 'Detections', 'rgba(25, 135, 84, 0.5)', 'rgba(25, 135, 84, 1)');
    const cpuUsageChart = createLineChart('cpuUsageChart', 'CPU Usage (%)', 'rgba(220, 53, 69, 0.5)', 'rgba(220, 53, 69, 1)');
    const memoryUsageChart = createLineChart('memoryUsageChart', 'Memory Usage (%)', 'rgba(255, 193, 7, 0.5)', 'rgba(255, 193, 7, 1)');
    
    // Initialize emotion distribution chart
    const emotionDistributionChart = createPieChart('emotionDistributionChart', [
        'Crying', 'Laughing', 'Babbling', 'Silence'
    ], [
        'rgba(220, 53, 69, 0.8)',
        'rgba(25, 135, 84, 0.8)',
        'rgba(13, 202, 240, 0.8)',
        'rgba(108, 117, 125, 0.8)'
    ]);
    
    // Initialize detection types chart
    const detectionTypesChart = createLineChart('detectionTypesChart', 'Detections', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 0, 1)');
    
    // Socket.IO connection for real-time updates
    const socket = io();
    
    // Track if socket is connected
    let socketConnected = false;
    
    // Connect to socket
    socket.on('connect', function() {
        console.log('Connected to metrics socket');
        socketConnected = true;
        connectionStatus.style.backgroundColor = '#198754'; // Green when connected
        connectionStatus.innerHTML = 'Socket: Connected';
        
        // Add to detection log
        addDetectionEvent(new Date().toLocaleString(), "Connected to server");
        
        // Request initial metrics data
        socket.emit('request_metrics', { timeRange: '1h' });
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from metrics socket');
        socketConnected = false;
        connectionStatus.style.backgroundColor = '#dc3545'; // Red when disconnected
        connectionStatus.innerHTML = 'Socket: Disconnected';
        
        // Add to detection log
        addDetectionEvent(new Date().toLocaleString(), "Disconnected from server", "warning");
    });
    
    socket.on('error', function(error) {
        console.error('Socket error:', error);
        connectionStatus.style.backgroundColor = '#dc3545'; // Red on error
        connectionStatus.innerHTML = 'Socket: Error';
        
        // Add to detection log
        addDetectionEvent(new Date().toLocaleString(), "Socket error: " + error, "error");
    });
    
    // Socket event for metrics updates
    socket.on('metrics_update', function(data) {
        console.log('Received metrics update:', data);
        connectionStatus.style.backgroundColor = '#0dcaf0'; // Blue when receiving data
        connectionStatus.innerHTML = 'Socket: Data Received';
        
        if (data && data.metrics) {
            updateCharts(data.metrics);
            
            // Update system information if available
            if (data.system_info) {
                updateSystemInfo(data.system_info);
            }
            
            // Update health bars
            updateHealthBars(data.metrics, data.system_info);
        } else if (data && data.current) {
            // Alternative data format with current data directly in the root
            const metricsData = {
                fps: Array(20).fill(0),
                detectionCount: Array(20).fill(0),
                cpuUsage: Array(20).fill(0),
                memoryUsage: Array(20).fill(0),
                detectionConfidence: Array(20).fill(0),
                emotions: {
                    crying: 0,
                    laughing: 0,
                    babbling: 0,
                    silence: 100
                }
            };
            
            // Update the last item with current values
            if (data.current.fps !== undefined) {
                metricsData.fps[19] = data.current.fps;
                document.getElementById('currentFps').textContent = data.current.fps.toFixed(1);
            }
            
            if (data.current.detections !== undefined) {
                metricsData.detectionCount[19] = data.current.detections;
                document.getElementById('currentDetections').textContent = data.current.detections;
                document.getElementById('personDetectionCount').textContent = data.current.detections;
            }
            
            if (data.current.cpu !== undefined) {
                metricsData.cpuUsage[19] = data.current.cpu;
                document.getElementById('currentCpu').textContent = data.current.cpu.toFixed(1) + '%';
            }
            
            if (data.current.memory !== undefined) {
                metricsData.memoryUsage[19] = data.current.memory;
                document.getElementById('currentMemory').textContent = data.current.memory.toFixed(1) + '%';
            }
            
            // Update system info if available
            if (data.system_info) {
                updateSystemInfo(data.system_info);
            }
            
            // Update health bars
            updateHealthBars(metricsData, data.system_info);
        } else {
            console.warn('Received metrics_update with invalid data format:', data);
            addDetectionEvent(new Date().toLocaleString(), "Received invalid data format", "warning");
        }
    });
    
    // Also listen for 'metrics' events (alternative format)
    socket.on('metrics', function(data) {
        console.log('Received metrics:', data);
        connectionStatus.style.backgroundColor = '#0dcaf0'; // Blue when receiving data
        connectionStatus.innerHTML = 'Socket: Data Received (metrics)';
        
        if (data) {
            const metricsData = {
                fps: Array(20).fill(0),
                detectionCount: Array(20).fill(0),
                cpuUsage: Array(20).fill(0),
                memoryUsage: Array(20).fill(0),
                detectionConfidence: Array(20).fill(0),
                emotions: {
                    crying: data.emotion_crying || 0,
                    laughing: data.emotion_laughing || 0,
                    babbling: data.emotion_babbling || 0,
                    silence: data.emotion_silence || 100
                }
            };
            
            // Update the last item with current values
            if (data.fps !== undefined) {
                metricsData.fps[19] = data.fps;
                document.getElementById('currentFps').textContent = data.fps.toFixed(1);
            }
            
            if (data.detection_count !== undefined) {
                metricsData.detectionCount[19] = data.detection_count;
                document.getElementById('currentDetections').textContent = data.detection_count;
                document.getElementById('personDetectionCount').textContent = data.detection_count;
            }
            
            if (data.cpu_usage !== undefined) {
                metricsData.cpuUsage[19] = data.cpu_usage;
                document.getElementById('currentCpu').textContent = data.cpu_usage.toFixed(1) + '%';
            }
            
            if (data.memory_usage !== undefined) {
                metricsData.memoryUsage[19] = data.memory_usage;
                document.getElementById('currentMemory').textContent = data.memory_usage.toFixed(1) + '%';
            }
            
            // Update emotion distributions
            updateEmotionDistribution(metricsData.emotions);
            
            // Update health bars
            updateHealthBars(metricsData, data);
        }
    });
    
    // Socket event for detection events
    socket.on('detection_event', function(data) {
        console.log('Detection event:', data);
        if (data && data.count) {
            addDetectionEvent(data.timestamp || new Date().toLocaleString(), `Detected ${data.count} person(s)`);
        }
    });
    
    // Function to update emotion distribution
    function updateEmotionDistribution(emotions) {
        if (!emotions) return;
        
        document.getElementById('cryingPercentage').textContent = emotions.crying.toFixed(1) + '%';
        document.getElementById('laughingPercentage').textContent = emotions.laughing.toFixed(1) + '%';
        document.getElementById('babblingPercentage').textContent = emotions.babbling.toFixed(1) + '%';
        document.getElementById('silencePercentage').textContent = emotions.silence.toFixed(1) + '%';
        
        emotionDistributionChart.data.datasets[0].data = [
            emotions.crying,
            emotions.laughing,
            emotions.babbling,
            emotions.silence
        ];
        emotionDistributionChart.update();
    }
    
    // Function to update charts with new data
    function updateCharts(data) {
        // Update FPS chart
        if (data.fps) {
            fpsChart.data.datasets[0].data = data.fps;
            fpsChart.update();
            document.getElementById('currentFps').textContent = data.fps[data.fps.length - 1].toFixed(1);
        }
        
        // Update Detection Count chart
        if (data.detectionCount) {
            detectionCountChart.data.datasets[0].data = data.detectionCount;
            detectionCountChart.update();
            document.getElementById('currentDetections').textContent = data.detectionCount[data.detectionCount.length - 1];
            document.getElementById('personDetectionCount').textContent = data.detectionCount[data.detectionCount.length - 1];
        }
        
        // Update Emotion Distribution chart
        if (data.emotions) {
            emotionDistributionChart.data.datasets[0].data = [
                data.emotions.crying,
                data.emotions.laughing,
                data.emotions.babbling,
                data.emotions.silence
            ];
            emotionDistributionChart.update();
            
            // Update percentages in the UI
            document.getElementById('cryingPercentage').textContent = data.emotions.crying.toFixed(1) + '%';
            document.getElementById('laughingPercentage').textContent = data.emotions.laughing.toFixed(1) + '%';
            document.getElementById('babblingPercentage').textContent = data.emotions.babbling.toFixed(1) + '%';
            document.getElementById('silencePercentage').textContent = data.emotions.silence.toFixed(1) + '%';
        }
        
        // Update Detection Types chart
        if (data.detectionConfidence) {
            detectionTypesChart.data.datasets[0].data = data.detectionConfidence;
            detectionTypesChart.update();
            
            // Update average confidence in the UI
            if (data.detectionConfidence.length > 0) {
                const avg = data.detectionConfidence.reduce((a, b) => a + b, 0) / data.detectionConfidence.length;
                document.getElementById('confidenceAvg').textContent = avg.toFixed(1) + '%';
            }
        }
        
        // Update CPU Usage chart
        if (data.cpuUsage) {
            cpuUsageChart.data.datasets[0].data = data.cpuUsage;
            cpuUsageChart.update();
            document.getElementById('currentCpu').textContent = data.cpuUsage[data.cpuUsage.length - 1] + '%';
        }
        
        // Update Memory Usage chart
        if (data.memoryUsage) {
            memoryUsageChart.data.datasets[0].data = data.memoryUsage;
            memoryUsageChart.update();
            document.getElementById('currentMemory').textContent = data.memoryUsage[data.memoryUsage.length - 1] + '%';
        }
    }
    
    // Update health bars based on metrics data
    function updateHealthBars(metrics, systemInfo) {
        // Update CPU health bar
        if (metrics.cpuUsage && metrics.cpuUsage.length > 0) {
            const cpuValue = metrics.cpuUsage[metrics.cpuUsage.length - 1];
            const cpuBar = document.getElementById('cpuHealthBar');
            const cpuValueEl = document.getElementById('cpuHealthValue');
            
            if (cpuBar && cpuValueEl) {
                cpuBar.style.width = `${cpuValue}%`;
                cpuValueEl.textContent = `${cpuValue.toFixed(1)}%`;
                
                if (cpuValue < 50) {
                    cpuBar.className = 'health-bar health-good';
                } else if (cpuValue < 80) {
                    cpuBar.className = 'health-bar health-warning';
                } else {
                    cpuBar.className = 'health-bar health-critical';
                }
            }
        }
        
        // Update Memory health bar
        if (metrics.memoryUsage && metrics.memoryUsage.length > 0) {
            const memValue = metrics.memoryUsage[metrics.memoryUsage.length - 1];
            const memBar = document.getElementById('memoryHealthBar');
            const memValueEl = document.getElementById('memoryHealthValue');
            
            if (memBar && memValueEl) {
                memBar.style.width = `${memValue}%`;
                memValueEl.textContent = `${memValue.toFixed(1)}%`;
                
                if (memValue < 50) {
                    memBar.className = 'health-bar health-good';
                } else if (memValue < 80) {
                    memBar.className = 'health-bar health-warning';
                } else {
                    memBar.className = 'health-bar health-critical';
                }
            }
        }
        
        // Update camera status badge
        if (systemInfo && systemInfo.camera_status) {
            const cameraStatusBadge = document.getElementById('cameraStatusBadge');
            if (cameraStatusBadge) {
                cameraStatusBadge.textContent = capitalizeFirstLetter(systemInfo.camera_status);
                cameraStatusBadge.className = `status-badge ${systemInfo.camera_status === 'running' || systemInfo.camera_status === 'connected' ? 'status-active' : 'status-inactive'}`;
            }
            
            // Update camera health bar
            const cameraHealthBar = document.getElementById('cameraHealthBar');
            if (cameraHealthBar) {
                if (systemInfo.camera_status === 'running' || systemInfo.camera_status === 'connected') {
                    cameraHealthBar.className = 'health-bar health-good';
                    cameraHealthBar.style.width = '100%';
                } else if (systemInfo.camera_status === 'initializing') {
                    cameraHealthBar.className = 'health-bar health-warning';
                    cameraHealthBar.style.width = '50%';
                } else {
                    cameraHealthBar.className = 'health-bar health-critical';
                    cameraHealthBar.style.width = '10%';
                }
            }
        }
        
        // Update AI status badge
        if (systemInfo && systemInfo.person_detector_status) {
            const aiStatusBadge = document.getElementById('aiStatusBadge');
            if (aiStatusBadge) {
                aiStatusBadge.textContent = capitalizeFirstLetter(systemInfo.person_detector_status);
                aiStatusBadge.className = `status-badge ${systemInfo.person_detector_status === 'running' ? 'status-active' : 'status-inactive'}`;
            }
            
            // Update AI health bar
            const aiHealthBar = document.getElementById('aiHealthBar');
            if (aiHealthBar) {
                if (systemInfo.person_detector_status === 'running') {
                    aiHealthBar.className = 'health-bar health-good';
                    aiHealthBar.style.width = '100%';
                } else if (systemInfo.person_detector_status === 'initializing') {
                    aiHealthBar.className = 'health-bar health-warning';
                    aiHealthBar.style.width = '50%';
                } else {
                    aiHealthBar.className = 'health-bar health-critical';
                    aiHealthBar.style.width = '10%';
                }
            }
        }
    }
    
    // Update system information display
    function updateSystemInfo(info) {
        if (!info) return;
        
        // Update each field if the element exists
        const updateField = (id, value) => {
            const element = document.getElementById(id);
            if (element && value !== undefined) {
                element.textContent = value;
            }
        };
        
        updateField('systemUptimeInfo', info.uptime || '00:00:00');
        updateField('personDetectorInfo', info.person_detector || 'YOLOv8 Active');
        updateField('detectorModelInfo', info.detector_model || 'YOLOv8n');
        updateField('detectionThreshold', info.detection_threshold || '0.7');
        updateField('emotionDetectorInfo', info.emotion_detector || 'Active');
        updateField('cameraResolution', info.camera_resolution || '640x480');
        updateField('audioSampleRate', info.audio_sample_rate || '16000 Hz');
        updateField('osInfo', info.os || 'Windows');
        updateField('pythonVersion', info.python_version || '3.8.10');
        updateField('opencvVersion', info.opencv_version || '4.5.4');
        
        // Update camera and microphone information (new fields)
        updateField('cameraDevice', info.camera_device || 'Default Camera');
        updateField('microphoneDevice', info.microphone_device || 'Default Microphone');
        
        // Update detection settings
        updateField('frameSkipValue', info.frame_skip || '2');
        updateField('processResolution', info.process_resolution || '640x480');
        updateField('confidenceThreshold', info.confidence_threshold || '0.7');
        updateField('detectionHistorySize', info.detection_history_size ? `${info.detection_history_size} frames` : '5 frames');
    }
    
    // Function to create a line chart
    function createLineChart(canvasId, label, backgroundColor, borderColor) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
        type: 'line',
        data: {
                labels: Array(20).fill(''),
            datasets: [{
                    label: label,
                    data: Array(20).fill(0),
                    backgroundColor: backgroundColor,
                    borderColor: borderColor,
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#aaaaaa'
                        }
                    },
                    x: {
                        display: false
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                animation: {
                    duration: 500
                }
            }
        });
    }
    
    // Function to create a pie chart
    function createPieChart(canvasId, labels, backgroundColors) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
        type: 'doughnut',
        data: {
                labels: labels,
            datasets: [{
                    data: Array(labels.length).fill(0),
                    backgroundColor: backgroundColors,
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
                            color: '#aaaaaa',
                        font: {
                                size: 10
                            }
                        }
                    }
                },
                animation: {
                    duration: 500
                }
            }
        });
    }
    
    // Add detection event to detection log
    function addDetectionEvent(timestamp, message, type = "info") {
        const detectionLog = document.getElementById('detectionLogContainer');
        if (!detectionLog) return;
        
        let icon = 'info-circle';
        let iconClass = 'text-info';
        
        if (type === "warning") {
            icon = 'exclamation-triangle';
            iconClass = 'text-warning';
        } else if (type === "error") {
            icon = 'exclamation-circle';
            iconClass = 'text-danger';
        } else if (type === "detection") {
            icon = 'person';
            iconClass = 'text-success';
        }
        
        const newEvent = document.createElement('div');
        newEvent.className = 'alert-item';
        newEvent.innerHTML = `
            <div class="alert-time">${timestamp}</div>
            <div class="alert-message">
                <i class="bi bi-${icon} ${iconClass}"></i> ${message}
            </div>
        `;
        detectionLog.insertBefore(newEvent, detectionLog.firstChild);
        
        // Limit to 10 events
        while (detectionLog.children.length > 10) {
            detectionLog.removeChild(detectionLog.lastChild);
        }
    }
    
    // Helper function to capitalize first letter
    function capitalizeFirstLetter(string) {
        if (!string) return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Time range buttons
    document.getElementById('timeRange1h').addEventListener('click', function() {
        this.classList.add('active');
        document.getElementById('timeRange3h').classList.remove('active');
        document.getElementById('timeRange24h').classList.remove('active');
        if (socketConnected) {
            addDetectionEvent(new Date().toLocaleString(), "Requesting 1h metrics data");
            socket.emit('request_metrics', { timeRange: '1h' });
        } else {
            addDetectionEvent(new Date().toLocaleString(), "Socket disconnected, cannot request data", "warning");
        }
    });
    
    document.getElementById('timeRange3h').addEventListener('click', function() {
        this.classList.add('active');
        document.getElementById('timeRange1h').classList.remove('active');
        document.getElementById('timeRange24h').classList.remove('active');
        if (socketConnected) {
            addDetectionEvent(new Date().toLocaleString(), "Requesting 3h metrics data");
            socket.emit('request_metrics', { timeRange: '3h' });
        } else {
            addDetectionEvent(new Date().toLocaleString(), "Socket disconnected, cannot request data", "warning");
        }
    });
    
    document.getElementById('timeRange24h').addEventListener('click', function() {
        this.classList.add('active');
        document.getElementById('timeRange1h').classList.remove('active');
        document.getElementById('timeRange3h').classList.remove('active');
        if (socketConnected) {
            addDetectionEvent(new Date().toLocaleString(), "Requesting 24h metrics data");
            socket.emit('request_metrics', { timeRange: '24h' });
        } else {
            addDetectionEvent(new Date().toLocaleString(), "Socket disconnected, cannot request data", "warning");
        }
    });
    
    // Clear detection log
    const clearLogBtn = document.getElementById('clearDetectionLog');
    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', function() {
            const logContainer = document.getElementById('detectionLogContainer');
            if (logContainer) {
                logContainer.innerHTML = '';
                addDetectionEvent(new Date().toLocaleString(), "Detection log cleared");
            }
        });
    }
    
    // Add demo mode toggle to the page
    const pageHeader = document.querySelector('.time-range-controls');
    if (pageHeader) {
        const demoToggle = document.createElement('button');
        demoToggle.id = 'demoModeToggle';
        demoToggle.className = 'btn btn-sm btn-outline-danger ml-2';
        demoToggle.style.marginLeft = '10px';
        demoToggle.innerHTML = 'Demo Mode';
        demoToggle.title = 'Toggle demo data generation for testing';
        pageHeader.appendChild(demoToggle);
        
        // Demo mode toggle
        let demoModeActive = false;
        let demoInterval = null;
        
        demoToggle.addEventListener('click', function() {
            demoModeActive = !demoModeActive;
            this.classList.toggle('active', demoModeActive);
            
            if (demoModeActive) {
                // Start demo data generation
                addDetectionEvent(new Date().toLocaleString(), "Demo mode activated");
                if (socketConnected) {
                    socket.emit('start_demo');
                }
                demoInterval = setInterval(generateRandomData, 2000);
            } else {
                // Stop demo data generation
                addDetectionEvent(new Date().toLocaleString(), "Demo mode deactivated");
                if (socketConnected) {
                    socket.emit('stop_demo');
                }
                if (demoInterval) {
                    clearInterval(demoInterval);
                    demoInterval = null;
                }
            }
        });
    }
    
    // Generate random data for demo mode
    function generateRandomData() {
        const randomFPS = Array(20).fill(0).map(() => Math.random() * 20 + 10);
        const randomDetections = Array(20).fill(0).map(() => Math.floor(Math.random() * 3));
        const randomConfidence = Array(20).fill(0).map(() => Math.random() * 30 + 70);
        const randomCPU = Array(20).fill(0).map(() => Math.random() * 50 + 10);
        const randomMemory = Array(20).fill(0).map(() => Math.random() * 40 + 20);
        
        const emotionData = {
            crying: Math.random() * 20,
            laughing: Math.random() * 20,
            babbling: Math.random() * 20
        };
        
        emotionData.silence = 100 - emotionData.crying - emotionData.laughing - emotionData.babbling;
        
        // Ensure non-negative values
        if (emotionData.silence < 0) {
            emotionData.silence = 0;
            const total = emotionData.crying + emotionData.laughing + emotionData.babbling;
            emotionData.crying = (emotionData.crying / total) * 100;
            emotionData.laughing = (emotionData.laughing / total) * 100;
            emotionData.babbling = (emotionData.babbling / total) * 100;
        }
        
        updateCharts({
            fps: randomFPS,
            detectionCount: randomDetections,
            detectionConfidence: randomConfidence,
            cpuUsage: randomCPU,
            memoryUsage: randomMemory,
            emotions: emotionData
        });
        
        // Update health bars with random data
        updateHealthBars({
            cpuUsage: randomCPU,
            memoryUsage: randomMemory
        }, {
            camera_status: Math.random() < 0.9 ? 'running' : 'error',
            person_detector_status: Math.random() < 0.9 ? 'running' : 'error'
        });
        
        // Add random detection events
        if (Math.random() < 0.3) { // 30% chance
            const count = Math.floor(Math.random() * 3) + 1;
            addDetectionEvent(new Date().toLocaleString(), `Demo: Detected ${count} person(s)`);
        }
    }
    
    // Initial chart update with empty data
    updateCharts({
        fps: Array(20).fill(0),
        detectionCount: Array(20).fill(0),
        detectionConfidence: Array(20).fill(0),
        cpuUsage: Array(20).fill(0),
        memoryUsage: Array(20).fill(0),
        emotions: {
            crying: 0,
            laughing: 0,
            babbling: 0,
            silence: 100
        }
    });
    
    // Add initial log message
    addDetectionEvent(new Date().toLocaleString(), "Metrics page initialized");
});
