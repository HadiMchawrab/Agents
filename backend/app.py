from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
import logging
from virtualgraph import test_graph

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'csv_test'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"


@app.route('/upload-and-process', methods=['POST'])
def upload_and_process():
    try:
        logger.debug("Received upload request")
        
        # Save uploaded files
        files = request.files.getlist('files')
        logger.debug(f"Received {len(files)} files")
        
        for file in files:
            if file.filename.endswith('.csv'):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                logger.debug(f"Saving file to {file_path}")
                file.save(file_path)

        # Get the initial state from the form data
        initialState = json.loads(request.form['initialState'])
        logger.debug(f"Initial state: {initialState}")
        
        # Convert the csv_files array back to a set
        initialState['csv_files'] = set(initialState['csv_files'])
        logger.debug(f"Converted csv_files to set: {initialState['csv_files']}")

        # Execute the graph with the initial state
        logger.debug("Starting graph execution")
        result = test_graph(initialState)
        logger.debug("Graph execution completed")

        return jsonify({
            'status': 'success',
            'message': 'Files processed successfully',
            'result': result
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # app.run(debug=True, port=8001)
    # Use the PORT environment variable or default to 8001
    # port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=5000)

# if __name__ == '__main__':
#     app.run(debug=True, port=8001) 