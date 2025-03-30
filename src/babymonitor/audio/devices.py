import pyaudio
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_available_microphones():
    """Get list of available microphones and current microphone.
    
    Returns:
        tuple: (list of microphones, current microphone info)
    """
    try:
        audio = pyaudio.PyAudio()
        microphones = []
        current_microphone = None
        
        # Get default input device
        try:
            default_device = audio.get_default_input_device_info()
            current_microphone = {
                'id': str(default_device['index']),
                'name': default_device['name'],
                'channels': default_device['maxInputChannels'],
                'sample_rate': int(default_device['defaultSampleRate'])
            }
        except Exception as e:
            logger.warning(f"Could not get default input device: {str(e)}")
        
        # Get all input devices
        for i in range(audio.get_device_count()):
            try:
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only input devices
                    microphones.append({
                        'id': str(device_info['index']),
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            except Exception as e:
                logger.warning(f"Error getting device info for index {i}: {str(e)}")
        
        return microphones, current_microphone
        
    except Exception as e:
        logger.error(f"Error getting microphones: {str(e)}")
        return [], None
    finally:
        if 'audio' in locals():
            audio.terminate()

def test_microphone(device_id, duration=0.5, sample_rate=44100):
    """Test a microphone by recording a short sample and analyzing it.
    
    Args:
        device_id: ID of the device to test
        duration (float): Duration in seconds to record
        sample_rate (int): Sample rate to use
        
    Returns:
        dict: Test results including status and measurements
    """
    try:
        audio = pyaudio.PyAudio()
        device_info = audio.get_device_info_by_index(int(device_id))
        
        # Record a short sample
        frames = []
        def callback(in_data, frame_count, time_info, status):
            frames.append(np.frombuffer(in_data, dtype=np.float32))
            return (in_data, pyaudio.paContinue)
        
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=int(device_id),
            frames_per_buffer=1024,
            stream_callback=callback
        )
        
        # Wait for the recording
        import time
        time.sleep(duration)
        
        # Stop and close
        stream.stop_stream()
        stream.close()
        
        # Analyze the recording
        if frames:
            audio_data = np.concatenate(frames)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            peak = np.max(np.abs(audio_data))
            
            # Convert to dB
            rms_db = 20 * np.log10(rms + 1e-10)
            peak_db = 20 * np.log10(peak + 1e-10)
            
            return {
                'status': 'success',
                'device_name': device_info['name'],
                'sample_rate': sample_rate,
                'channels': device_info['maxInputChannels'],
                'rms_level': float(rms_db),
                'peak_level': float(peak_db)
            }
        else:
            return {
                'status': 'error',
                'error': 'No audio data received'
            }
            
    except Exception as e:
        logger.error(f"Error testing microphone: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        if 'audio' in locals():
            audio.terminate() 