from flask import Blueprint, jsonify, request
from ...detectors import EmotionDetector

bp = Blueprint('audio', __name__)
emotion_detector = None

def init_app(app, detector):
    global emotion_detector
    emotion_detector = detector
    app.register_blueprint(bp, url_prefix='/api/audio')

@bp.route('/microphone', methods=['GET'])
def get_microphone_status():
    """Get current microphone status."""
    if emotion_detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Emotion detector not initialized'
        }), 500
        
    status = emotion_detector.get_microphone_status()
    return jsonify(status)

@bp.route('/level', methods=['GET'])
def get_audio_level():
    """Get current audio input level."""
    if emotion_detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Emotion detector not initialized',
            'level': -60.0
        }), 500
        
    level = emotion_detector.get_audio_level()
    return jsonify({
        'status': 'success',
        'level': level
    })

@bp.route('/microphones', methods=['GET'])
def get_available_microphones():
    """Get list of available microphones."""
    import sounddevice as sd
    try:
        devices = sd.query_devices()
        microphones = []
        
        for device in devices:
            if device['max_input_channels'] > 0:
                microphones.append({
                    'id': device['index'],
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default': device['index'] == sd.default.device[0]
                })
                
        current_mic = emotion_detector.current_microphone_id if emotion_detector else None
        
        return jsonify({
            'status': 'success',
            'microphones': microphones,
            'current_microphone': current_mic
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'microphones': []
        }), 500

@bp.route('/test', methods=['POST'])
def test_microphone():
    """Test microphone by attempting to initialize it."""
    if emotion_detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Emotion detector not initialized'
        }), 500
        
    try:
        data = request.get_json()
        microphone_id = data.get('microphone_id')
        
        if microphone_id is None:
            return jsonify({
                'status': 'error',
                'message': 'No microphone ID provided'
            }), 400
            
        success = emotion_detector.set_microphone(str(microphone_id))
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Microphone test successful'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Microphone test failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 