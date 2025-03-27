from flask import Flask, render_template
from api_routes import api_bp
import os

# Create Flask app
app = Flask(__name__)

# Register API blueprint
app.register_blueprint(api_bp)

# Base route
@app.route('/')
def home():
    return render_template('index.html')

# Repair tools route
@app.route('/repair_tools')
def repair_tools():
    mode = "normal"
    dev_mode = False
    return render_template('repair_tools.html', mode=mode, dev_mode=dev_mode)

# Developer mode repair tools route
@app.route('/repair_tools/dev')
def repair_tools_dev():
    mode = "dev"
    dev_mode = True
    return render_template('repair_tools.html', mode=mode, dev_mode=dev_mode)

if __name__ == '__main__':
    # Ensure template folder structure is set correctly
    app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'templates')
    app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'static')
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True) 