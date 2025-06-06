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
        socket.emit('request_metrics', { timeRange: currentTimeRange });
    });

    // Current selected time range
    let currentTimeRange = '1h';

    // Time range control event listeners
    document.getElementById('timeRange1h').addEventListener('click', function() {
        setTimeRange('1h');
    });
    
    document.getElementById('timeRange3h').addEventListener('click', function() {
        setTimeRange('3h');
    });
    
    document.getElementById('timeRange24h').addEventListener('click', function() {
        setTimeRange('24h');
    });

    function setTimeRange(range) {
        // Update buttons UI
        document.getElementById('timeRange1h').classList.remove('active');
        document.getElementById('timeRange3h').classList.remove('active');
        document.getElementById('timeRange24h').classList.remove('active');
        
        document.getElementById('timeRange' + range).classList.add('active');
        
        currentTimeRange = range;
        
        // Request metrics with new time range
        if (socketConnected) {
            socket.emit('request_metrics', { timeRange: range });
        }
    }

    // Initialize charts
    const fpsChart = createLineChart('fpsChart', 'FPS', 'rgba(13, 202, 240, 0.5)', 'rgba(13, 202, 240, 1)');
    const detectionCountChart = createLineChart('detectionCountChart', 'Detections', 'rgba(25, 135, 84, 0.5)', 'rgba(25, 135, 84, 1)');
    const cpuUsageChart = createLineChart('cpuUsageChart', 'CPU Usage (%)', 'rgba(220, 53, 69, 0.5)', 'rgba(220, 53, 69, 1)');
    const memoryUsageChart = createLineChart('memoryUsageChart', 'Memory Usage (%)', 'rgba(255, 193, 7, 0.5)', 'rgba(255, 193, 7, 1)');
    
    // Default emotion labels and colors
    const defaultEmotionLabels = ['Crying', 'Laughing', 'Babbling', 'Silence'];
    const defaultEmotionColors = [
        'rgba(220, 53, 69, 0.8)',   // Crying - Red
        'rgba(25, 135, 84, 0.8)',   // Laughing - Green
        'rgba(13, 202, 240, 0.8)',  // Babbling - Blue
        'rgba(108, 117, 125, 0.8)'  // Silence - Gray
    ];
    
    // Mapping for emotion colors
    const emotionColorMap = {
        'crying': 'rgba(220, 53, 69, 0.8)',
        'laughing': 'rgba(25, 135, 84, 0.8)',
        'babbling': 'rgba(13, 202, 240, 0.8)',
        'silence': 'rgba(108, 117, 125, 0.8)',
        'happy': 'rgba(25, 135, 84, 0.8)',
        'sad': 'rgba(220, 53, 69, 0.8)',
        'angry': 'rgba(255, 193, 7, 0.8)',
        'neutral': 'rgba(108, 117, 125, 0.8)',
        'not_crying': 'rgba(25, 135, 84, 0.8)'
    };
    
    // Initialize emotion distribution chart
    let emotionDistributionChart = createPieChart('emotionDistributionChart', defaultEmotionLabels, defaultEmotionColors);
    
    // Current emotion model info
    let currentEmotionModel = {
        id: 'unknown',
        name: 'Unknown Model',
        emotions: defaultEmotionLabels.map(l => l.toLowerCase())
    };
    
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
        socket.emit('request_metrics', { timeRange: currentTimeRange });
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
        
        if (data && data.current) {
            // Update current metric values
            if (data.current.fps !== undefined) {
                document.getElementById('currentFps').textContent = data.current.fps.toFixed(1);
            }
            
            if (data.current.detection_count !== undefined) {
                document.getElementById('currentDetections').textContent = data.current.detection_count;
            }
            
            if (data.current.cpu_usage !== undefined) {
                document.getElementById('currentCpu').textContent = data.current.cpu_usage.toFixed(1) + '%';
            }
            
            if (data.current.memory_usage !== undefined) {
                document.getElementById('currentMemory').textContent = data.current.memory_usage.toFixed(1) + '%';
            }
            
            // Update emotion related displays
            if (data.current.emotion !== undefined) {
                updateCurrentEmotion(data.current.emotion, data.current.emotion_confidence || 0);
            }
            
            // Update emotion distribution if available
            if (data.history && data.history.emotions) {
                updateEmotionDistribution(data.history.emotions);
            }
            
            // Update health bars
            updateHealthBars(data);
        }
    });
    
    // Socket event for emotion history updates
    socket.on('emotion_history', function(data) {
        console.log('Received emotion history:', data);
        if (data && data.percentages) {
            updateEmotionDistribution(data.percentages);
        }
    });
    
    // Socket event for emotion model changes
    socket.on('emotion_model_changed', function(modelInfo) {
        console.log('Emotion model changed:', modelInfo);
        
        // Update current model info
        currentEmotionModel = modelInfo;
        
        // Recreate emotion chart with new emotions
        updateEmotionChartForModel(modelInfo);
        
        // Add model change to log
        addDetectionEvent(
            new Date().toLocaleString(), 
            `Emotion model changed to ${modelInfo.name}`
        );
    });
    
    // Socket event for emotion updates
    socket.on('emotion_update', function(data) {
        console.log('Emotion update:', data);
        
        if (data && data.emotion) {
            // Update current emotion displays
            updateCurrentEmotion(data.emotion, data.confidence || 0);
            
            // Add to log if confidence is high enough
            if (data.confidence > 0.7) {
                let emotionMessage = getEmotionMessage(data.emotion, data.confidence);
                addDetectionEvent(new Date().toLocaleString(), emotionMessage);
            }
        }
    });
    
    // Function to get emotion message
    function getEmotionMessage(emotion, confidence) {
        if (emotion === 'crying') {
            if (confidence > 0.85) {
                return "Baby is crying loudly! Needs immediate attention.";
            } else if (confidence > 0.7) {
                return "Baby is crying. May need attention.";
            } else {
                return "Baby might be starting to cry.";
            }
        } else if (emotion === 'laughing') {
            if (confidence > 0.8) {
                return "Baby is happily laughing!";
            } else {
                return "Baby is making happy sounds.";
            }
        } else if (emotion === 'babbling') {
            return "Baby is babbling or talking.";
        } else if (emotion === 'silence') {
            return "Baby is quiet.";
        } else {
            return `Detected ${emotion} (${(confidence * 100).toFixed(0)}% confidence)`;
        }
    }
    
    // Socket event for detection events
    socket.on('detection_update', function(data) {
        console.log('Detection event:', data);
        if (data && data.count > 0) {
            addDetectionEvent(new Date().toLocaleString(), `Detected ${data.count} person(s)`);
        }
    });
    
    // Socket event for alerts
    socket.on('alert', function(alert) {
        console.log('Alert received:', alert);
        
        if (alert.type === 'crying') {
            // Add crying alert with high priority
            addDetectionEvent(new Date().toLocaleString(), alert.message, "danger");
            
            // Update emotion in UI if needed
            updateCurrentEmotion('crying', 0.9);
        }
    });
    
    // Function to update the emotion chart based on the model
    function updateEmotionChartForModel(modelInfo) {
        // Get emotions from model
        const emotions = modelInfo.emotions || defaultEmotionLabels.map(l => l.toLowerCase());
        
        // Create labels with capitalized first letter
        const labels = emotions.map(e => capitalizeFirstLetter(e));
        
        // Create colors array based on emotions
        const colors = emotions.map(e => emotionColorMap[e] || 'rgba(108, 117, 125, 0.8)');
        
        // Destroy existing chart if it exists
        if (emotionDistributionChart) {
            emotionDistributionChart.destroy();
        }
        
        // Create new chart with model-specific emotions
        emotionDistributionChart = createPieChart('emotionDistributionChart', labels, colors);
        
        // Update emotion percentage containers
        updateEmotionPercentageContainers(emotions);
    }
    
    // Function to update emotion percentage display containers
    function updateEmotionPercentageContainers(emotions) {
        // Get the container
        const container = document.querySelector('.emotion-distribution');
        if (!container) return;
        
        // Clear the container
        container.innerHTML = '';
        
        // Create emotion items for each emotion
        emotions.forEach(emotion => {
            // Create emotion item
            const emotionItem = document.createElement('div');
            emotionItem.className = `emotion-item emotion-${emotion}`;
            
            // Add icon based on emotion
            let iconClass = 'bi-emoji-neutral';
            if (emotion === 'crying') iconClass = 'bi-emoji-frown';
            else if (emotion === 'laughing' || emotion === 'happy') iconClass = 'bi-emoji-laughing';
            else if (emotion === 'babbling') iconClass = 'bi-chat-dots';
            else if (emotion === 'silence') iconClass = 'bi-volume-mute';
            else if (emotion === 'sad') iconClass = 'bi-emoji-frown';
            else if (emotion === 'angry') iconClass = 'bi-emoji-angry';
            
            // Create icon div
            const iconDiv = document.createElement('div');
            iconDiv.className = 'emotion-icon';
            iconDiv.innerHTML = `<i class="bi ${iconClass}"></i>`;
            
            // Create value div
            const valueDiv = document.createElement('div');
            valueDiv.className = 'emotion-value';
            valueDiv.id = `${emotion}Percentage`;
            valueDiv.textContent = '0%';
            
            // Create label div
            const labelDiv = document.createElement('div');
            labelDiv.className = 'emotion-label';
            labelDiv.textContent = capitalizeFirstLetter(emotion);
            
            // Add all elements to the item
            emotionItem.appendChild(iconDiv);
            emotionItem.appendChild(valueDiv);
            emotionItem.appendChild(labelDiv);
            
            // Add the item to the container
            container.appendChild(emotionItem);
        });
        }
        
    // Function to update current emotion display
    function updateCurrentEmotion(emotion, confidence) {
        // Update normal mode display if it exists
        if (typeof appMode !== 'undefined' && appMode === 'normal') {
            const normalEmotion = document.getElementById('normalEmotion');
            if (normalEmotion) {
                let displayEmotion = capitalizeFirstLetter(emotion);
                if (emotion === 'not_crying') displayEmotion = 'Not Crying';
                normalEmotion.textContent = displayEmotion;
            }
        }
        
        // Update any other emotion displays
        // For example, add a visual indicator or sound for high confidence crying
        if (emotion === 'crying' && confidence > 0.8) {
            // Flash the display or play a sound
            if (document.getElementById('cryingPercentage')) {
                document.getElementById('cryingPercentage').style.animation = 'flash 1s infinite';
                setTimeout(() => {
                    if (document.getElementById('cryingPercentage')) {
                        document.getElementById('cryingPercentage').style.animation = '';
                    }
                }, 5000);
            }
        }
        }
        
    // Function to update emotion distribution
    function updateEmotionDistribution(emotions) {
        if (!emotions) return;
        
        // Get all emotion percentages
        const currentModelEmotions = currentEmotionModel.emotions || defaultEmotionLabels.map(e => e.toLowerCase());
            
        // Update percentage displays for all emotions
        currentModelEmotions.forEach(emotion => {
            const percentageElement = document.getElementById(`${emotion}Percentage`);
            if (percentageElement) {
                const percentage = emotions[emotion] !== undefined ? emotions[emotion] : 0;
                percentageElement.textContent = percentage.toFixed(1) + '%';
            }
        });
        
        // Update chart data
        if (emotionDistributionChart && emotionDistributionChart.data) {
            const data = currentModelEmotions.map(emotion => {
                return emotions[emotion] !== undefined ? emotions[emotion] : 0;
            });
            
            emotionDistributionChart.data.datasets[0].data = data;
            emotionDistributionChart.update();
        }
    }
    
    // Function to update health bars based on metrics
    function updateHealthBars(data) {
        // CPU Usage
        if (data.current && data.current.cpu_usage !== undefined) {
            const cpuUsage = data.current.cpu_usage;
            const cpuBar = document.getElementById('cpuHealthBar');
            const cpuValue = document.getElementById('cpuHealthValue');
            
            if (cpuBar && cpuValue) {
                cpuValue.textContent = cpuUsage.toFixed(1) + '%';
                cpuBar.style.width = cpuUsage + '%';
                
                cpuBar.className = 'health-bar';
                if (cpuUsage > 80) {
                    cpuBar.classList.add('health-critical');
                } else if (cpuUsage > 60) {
                    cpuBar.classList.add('health-warning');
                } else {
                    cpuBar.classList.add('health-good');
                }
            }
        }
        
        // Memory Usage
        if (data.current && data.current.memory_usage !== undefined) {
            const memUsage = data.current.memory_usage;
            const memBar = document.getElementById('memoryHealthBar');
            const memValue = document.getElementById('memoryHealthValue');
            
            if (memBar && memValue) {
                memValue.textContent = memUsage.toFixed(1) + '%';
                memBar.style.width = memUsage + '%';
                
                memBar.className = 'health-bar';
                if (memUsage > 80) {
                    memBar.classList.add('health-critical');
                } else if (memUsage > 60) {
                    memBar.classList.add('health-warning');
                } else {
                    memBar.classList.add('health-good');
                }
            }
        }
        
        // Camera Status
        const cameraBar = document.getElementById('cameraHealthBar');
        const cameraStatus = document.getElementById('cameraStatusBadge');
        
        if (cameraBar && cameraStatus) {
            if (data.camera_status === 'connected') {
                cameraBar.className = 'health-bar health-good';
                cameraBar.style.width = '100%';
                cameraStatus.className = 'status-badge status-active';
                cameraStatus.textContent = 'Active';
                } else {
                cameraBar.className = 'health-bar health-critical';
                cameraBar.style.width = '100%';
                cameraStatus.className = 'status-badge status-inactive';
                cameraStatus.textContent = 'Inactive';
            }
        }
        
        // AI Processing Status
        const aiBar = document.getElementById('aiHealthBar');
        const aiStatus = document.getElementById('aiStatusBadge');
        
        if (aiBar && aiStatus) {
            if (data.emotion_detector_status === 'running') {
                aiBar.className = 'health-bar health-good';
                aiBar.style.width = '100%';
                aiStatus.className = 'status-badge status-active';
                aiStatus.textContent = 'Active';
                } else {
                aiBar.className = 'health-bar health-critical';
                aiBar.style.width = '100%';
                aiStatus.className = 'status-badge status-inactive';
                aiStatus.textContent = 'Inactive';
            }
        }
    }
    
    // Function to format emotion data for charts
    function formatEmotionData(emotions) {
        // Default to empty values if no data provided
        if (!emotions) {
            return {
                labels: [],
                data: []
            };
        }

        const labels = Object.keys(emotions).map(key => capitalizeFirstLetter(key));
        const data = Object.values(emotions);
        
        return { labels, data };
    }
    
    // Function to create a line chart
    function createLineChart(canvasId, label, backgroundColor, borderColor) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        
        const chart = new Chart(ctx, {
        type: 'line',
        data: {
                labels: Array(20).fill(''),
            datasets: [{
                    label: label,
                    data: Array(20).fill(0),
                    backgroundColor: backgroundColor,
                    borderColor: borderColor,
                borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
            }]
        },
            options: {
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true
                    }
                },
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
                        grid: {
                            display: false
                        },
                        ticks: {
                        display: false
                    }
                    }
                },
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 500
                }
            }
        });
        
        return chart;
    }
    
    // Function to create a pie chart
    function createPieChart(canvasId, labels, backgroundColors) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        
        const chart = new Chart(ctx, {
            type: 'pie',
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
            plugins: {
                legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.raw.toFixed(1) + '%';
                            }
                        }
                    }
                },
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 500
                }
            }
        });
        
        return chart;
    }
    
    // Function to add an event to the detection log
    function addDetectionEvent(timestamp, message, type = "info") {
        const container = document.getElementById('detectionLogContainer');
        if (!container) return;
        
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item';
        
        // Add class based on type
        if (type === 'danger' || type === 'error') {
            alertItem.classList.add('alert-danger');
        } else if (type === 'warning') {
            alertItem.classList.add('alert-warning');
        } else {
            alertItem.classList.add('alert-info');
        }
        
        const alertTime = document.createElement('div');
        alertTime.className = 'alert-time';
        alertTime.textContent = timestamp;
        
        const alertMessage = document.createElement('div');
        alertMessage.className = 'alert-message';
        
        // Add icon based on type
        let iconClass = 'bi-info-circle text-info';
        if (type === 'danger' || type === 'error') {
            iconClass = 'bi-exclamation-triangle text-danger';
        } else if (type === 'warning') {
            iconClass = 'bi-exclamation-circle text-warning';
        }
        
        alertMessage.innerHTML = `<i class="${iconClass}"></i> ${message}`;
        
        alertItem.appendChild(alertTime);
        alertItem.appendChild(alertMessage);
        
        // Add to the top of the container
        container.insertBefore(alertItem, container.firstChild);
        
        // Limit the number of items
        const maxItems = 20;
        while (container.children.length > maxItems) {
            container.removeChild(container.lastChild);
        }
    }
    
    // Helper function to capitalize first letter
    function capitalizeFirstLetter(string) {
        if (!string) return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Fetch initial data and set up emotion model
    fetch('/api/system_info')
        .then(response => response.json())
        .then(data => {
            console.log('Fetched system info:', data);
            
            // Initialize UI based on system info
            if (data.emotion_model) {
                currentEmotionModel = data.emotion_model;
                updateEmotionChartForModel(data.emotion_model);
            }
            
            // Update system health bars
            updateHealthBars(data);
        })
        .catch(error => {
            console.error('Error fetching system info:', error);
    });
    
    // Fetch emotion log entries
    fetch('/api/emotion_log')
        .then(response => response.json())
        .then(data => {
            console.log('Fetched emotion log:', data);
            
            if (data.status === 'success' && data.log && data.log.length > 0) {
                // Display log entries
                data.log.slice().reverse().forEach(entry => {
                    if (entry.timestamp) {
                        const timestamp = new Date(entry.timestamp * 1000).toLocaleString();
                        let message = entry.message;
        
                        // If it's a model change entry
                        if (entry.type === 'model_changed') {
                            message = `Emotion model changed to ${entry.model}`;
                        }
                        
                        addDetectionEvent(timestamp, message || `Detected ${entry.emotion}`);
            }
        });
    }
        })
        .catch(error => {
            console.error('Error fetching emotion log:', error);
        });
        
    // If we're in normal mode, set up polling for updates
    if (typeof appMode !== 'undefined' && appMode === 'normal') {
        setInterval(() => {
            if (socketConnected) {
                // Request updated metrics
                socket.emit('request_metrics', { timeRange: currentTimeRange });
            }
        }, 10000); // Poll every 10 seconds
    }
});
