"""
API Endpoint Definitions
=======================
Centralizes the API endpoint URLs to ensure consistency between frontend and backend.
This helps prevent 404 errors due to URL mismatches.
"""

# Standard API URL prefixes
API_PREFIX = '/api'

# Emotion detection endpoints
EMOTION_PREFIX = f'{API_PREFIX}/emotion'
EMOTION_MODELS = f'{EMOTION_PREFIX}/models'
EMOTION_MODEL_INFO = f'{EMOTION_PREFIX}/model_info'  # Will be used with /<model_id>
EMOTION_SWITCH_MODEL = f'{EMOTION_PREFIX}/switch_model'
EMOTION_TEST_AUDIO = f'{EMOTION_PREFIX}/test_audio'
EMOTION_RESTART_AUDIO = f'{EMOTION_PREFIX}/restart_audio'

# Camera endpoints
CAMERA_PREFIX = f'{API_PREFIX}/cameras'
CAMERA_LIST = CAMERA_PREFIX
CAMERA_ADD = f'{CAMERA_PREFIX}/add'
CAMERA_REMOVE = f'{CAMERA_PREFIX}/remove'
CAMERA_ACTIVATE = f'{CAMERA_PREFIX}/activate'
CAMERA_FEED = f'{CAMERA_PREFIX}/<camera_id>/feed'

# System endpoints
SYSTEM_PREFIX = f'{API_PREFIX}/system'
SYSTEM_CHECK = f'{SYSTEM_PREFIX}/check'
SYSTEM_INFO = f'{SYSTEM_PREFIX}/info'
SYSTEM_RESTART = f'{SYSTEM_PREFIX}/restart'
SYSTEM_STOP = f'{SYSTEM_PREFIX}/stop'

# Legacy endpoints (for backward compatibility)
AUDIO_TEST = f'{API_PREFIX}/audio/test'
AUDIO_RESTART = f'{API_PREFIX}/audio/restart'
CAMERA_OLD_ADD = f'{API_PREFIX}/camera/add'
CAMERA_OLD_REMOVE = f'{API_PREFIX}/camera/remove'
CAMERA_OLD_RESTART = f'{API_PREFIX}/camera/restart'

# Other routes
VIDEO_FEED = '/video_feed'
METRICS_PAGE = '/metrics'
REPAIR_TOOLS = '/repair'
INDEX_PAGE = '/'
FAVICON = '/favicon.ico'

def generate_js_constants():
    """Generate JavaScript constants for the API endpoints."""
    return f"""
// API Endpoints - Auto-generated
const API = {{
    // Emotion endpoints
    emotion: {{
        models: '{EMOTION_MODELS}',
        modelInfo: '{EMOTION_MODEL_INFO}',  // Use with + '/' + modelId
        switchModel: '{EMOTION_SWITCH_MODEL}',
        testAudio: '{EMOTION_TEST_AUDIO}',
        restartAudio: '{EMOTION_RESTART_AUDIO}'
    }},
    
    // Camera endpoints
    cameras: {{
        list: '{CAMERA_LIST}',
        add: '{CAMERA_ADD}',
        remove: '{CAMERA_REMOVE}',
        activate: '{CAMERA_ACTIVATE}',
        feed: '{CAMERA_PREFIX}/'  // Use with + cameraId + '/feed'
    }},
    
    // System endpoints
    system: {{
        check: '{SYSTEM_CHECK}',
        info: '{SYSTEM_INFO}',
        restart: '{SYSTEM_RESTART}',
        stop: '{SYSTEM_STOP}'
    }}
}};
""" 