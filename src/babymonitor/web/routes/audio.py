from flask import Blueprint, jsonify, request
import numpy as np
import logging
from ...audio.devices import get_available_microphones, test_microphone

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
bp = Blueprint('audio', __name__)

@bp.route('/microphones', methods=['GET'])
def get_microphones():
    """Get list of available microphones."""
    try:
        microphones, current_microphone = get_available_microphones()
        logger.info(f"Found {len(microphones)} microphones")
        return jsonify({
            "microphones": microphones,
            "current_microphone": current_microphone
        })
    except Exception as e:
        logger.error(f"Error getting microphones: {str(e)}")
        return jsonify({
            "error": "Failed to get microphones",
            "message": str(e)
        }), 500

@bp.route('/microphone_info/<microphone_id>', methods=['GET'])
def get_microphone_info(microphone_id):
    """Get detailed information about a specific microphone."""
    try:
        info = test_microphone(microphone_id)
        if info['status'] == 'error':
            return jsonify(info), 400
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting microphone info: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@bp.route('/set_microphone', methods=['POST'])
def set_microphone():
    """Set the current microphone."""
    try:
        data = request.get_json()
        if not data or 'microphone_id' not in data:
            return jsonify({
                "status": "error",
                "message": "No microphone_id provided"
            }), 400

        microphone_id = data['microphone_id']
        # Test the microphone first
        test_result = test_microphone(microphone_id)
        if test_result['status'] == 'error':
            return jsonify({
                "status": "error",
                "message": f"Microphone test failed: {test_result['error']}"
            }), 400

        return jsonify({
            "status": "success",
            "message": "Microphone selection applied successfully",
            "microphone_info": test_result
        })
    except Exception as e:
        logger.error(f"Error setting microphone: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@bp.route('/level', methods=['GET'])
def get_audio_level():
    """Get the current audio level from the microphone."""
    try:
        microphone_id = request.args.get('microphone_id')
        if not microphone_id:
            return jsonify({'error': 'No microphone ID provided'}), 400

        # For now, return a simulated audio level
        # This will be replaced with actual audio level monitoring
        return jsonify({'level': -30})  # Simulated moderate level
    except Exception as e:
        logger.error(f"Error getting audio level: {str(e)}")
        return jsonify({'error': str(e)}), 500 