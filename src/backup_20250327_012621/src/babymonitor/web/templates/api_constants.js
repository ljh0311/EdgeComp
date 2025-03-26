// API Endpoints
const API = {
    // Emotion endpoints
    emotion: {
        models: '/api/emotion/models',
        modelInfo: '/api/emotion/model_info',  // Use with + '/' + modelId
        switchModel: '/api/emotion/switch_model',
        testAudio: '/api/emotion/test_audio',
        restartAudio: '/api/emotion/restart_audio'
    },
    
    // Camera endpoints
    cameras: {
        list: '/api/cameras',
        add: '/api/cameras/add',
        remove: '/api/cameras/remove',
        activate: '/api/cameras/activate',
        feed: '/api/cameras/'  // Use with + cameraId + '/feed'
    },
    
    // System endpoints
    system: {
        check: '/api/system/check',
        info: '/api/system/info',
        restart: '/api/system/restart',
        stop: '/api/system/stop'
    }
}; 