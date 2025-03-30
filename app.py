from flask import Flask, jsonify, render_template
from src.babymonitor.web.routes import audio
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    # Set up template directory
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'babymonitor', 'web', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'babymonitor', 'web', 'static'))
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Register blueprints
    app.register_blueprint(audio.bp, url_prefix='/api/audio')
    
    @app.route('/')
    def home():
        return render_template('repair_tools.html', mode="normal", dev_mode=False)
    
    @app.route('/repair_tools')
    def repair_tools():
        return render_template('repair_tools.html', mode="normal", dev_mode=False)
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not Found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal Server Error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    try:
        logger.info("Starting Baby Monitor web server...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}") 