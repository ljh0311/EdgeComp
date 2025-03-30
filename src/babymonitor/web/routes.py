"""
API Routes
---------
Defines all API routes for the Baby Monitor System.
"""

# Basic pages
INDEX_PAGE = "/"
METRICS_PAGE = "/metrics"
REPAIR_TOOLS = "/repair"

# API endpoints
API_PREFIX = "/api"

# System endpoints
SYSTEM_STATUS = f"{API_PREFIX}/system/status"
SYSTEM_INFO = f"{API_PREFIX}/system/info"
SYSTEM_RESTART = f"{API_PREFIX}/system/restart"
SYSTEM_STOP = f"{API_PREFIX}/system/stop"

# Camera endpoints
CAMERA_LIST = f"{API_PREFIX}/cameras"
CAMERA_ADD = f"{API_PREFIX}/cameras/add"
CAMERA_REMOVE = f"{API_PREFIX}/cameras/remove/<camera_id>"
CAMERA_ACTIVATE = f"{API_PREFIX}/cameras/activate/<camera_id>"
VIDEO_FEED = f"{API_PREFIX}/video_feed"

# Audio endpoints
AUDIO_MICROPHONES = f"{API_PREFIX}/audio/microphones"
AUDIO_SET_MICROPHONE = f"{API_PREFIX}/audio/set_microphone"
AUDIO_LEVEL = f"{API_PREFIX}/audio/level"
AUDIO_INFO = f"{API_PREFIX}/audio/microphone_info"
AUDIO_TEST = f"{API_PREFIX}/audio/test"
AUDIO_RESTART = f"{API_PREFIX}/audio/restart"

# Emotion detection endpoints
EMOTION_MODELS = f"{API_PREFIX}/emotion/models"
EMOTION_MODEL_INFO = f"{API_PREFIX}/emotion/model_info"
EMOTION_SWITCH_MODEL = f"{API_PREFIX}/emotion/switch_model"
EMOTION_TEST_AUDIO = f"{API_PREFIX}/emotion/test_audio"
EMOTION_RESTART_AUDIO = f"{API_PREFIX}/emotion/restart_audio"

def generate_js_constants():
    """Generate JavaScript constants for the API endpoints."""
    return f"""
// API Endpoints - Auto-generated
const API = {{
    // System endpoints
    system: {{
        status: '{SYSTEM_STATUS}',
        info: '{SYSTEM_INFO}',
        restart: '{SYSTEM_RESTART}',
        stop: '{SYSTEM_STOP}'
    }},
    
    // Camera endpoints
    cameras: {{
        list: '{CAMERA_LIST}',
        add: '{CAMERA_ADD}',
        remove: '{CAMERA_REMOVE.replace("<camera_id>", "")}',
        activate: '{CAMERA_ACTIVATE.replace("<camera_id>", "")}',
        feed: '{VIDEO_FEED}'
    }},
    
    // Audio endpoints
    audio: {{
        microphones: '{AUDIO_MICROPHONES}',
        setMicrophone: '{AUDIO_SET_MICROPHONE}',
        level: '{AUDIO_LEVEL}',
        info: '{AUDIO_INFO}',
        test: '{AUDIO_TEST}',
        restart: '{AUDIO_RESTART}'
    }},
    
    // Emotion endpoints
    emotion: {{
        models: '{EMOTION_MODELS}',
        modelInfo: '{EMOTION_MODEL_INFO}',
        switchModel: '{EMOTION_SWITCH_MODEL}',
        testAudio: '{EMOTION_TEST_AUDIO}',
        restartAudio: '{EMOTION_RESTART_AUDIO}'
    }}
}};
""" 