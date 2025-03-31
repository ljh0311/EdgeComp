from flask import Flask, jsonify, render_template, request
from api_routes import api_bp
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
    app.register_blueprint(api_bp)  # This includes all API routes
    
    @app.route('/')
    def home():
        return render_template('index.html', mode="normal", dev_mode=False)
    
    @app.route('/repair')
    def repair_tools():
        return render_template('repair_tools.html', mode="normal", dev_mode=False)
    
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not Found', 'path': request.path}), 404
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal Server Error'}), 500
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    try:
        logger.info("Starting Baby Monitor web server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}") 