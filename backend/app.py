from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from virtualgraph import test_graph

app = Flask(__name__)
CORS(app)
    
UPLOAD_FOLDER = 'csv_test'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"

<<<<<<< HEAD

@app.route('/upload-and-process', methods=['POST'])
def upload_and_process():
    try:
        # Save uploaded files
        files = request.files.getlist('files')
        for file in files:
            if file.filename.endswith('.csv'):
                file.save(os.path.join(UPLOAD_FOLDER, file.filename))
=======
# Configure minimal console-only logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)
>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565

        # Get the initial state from the form data
        initialState = json.loads(request.form['initialState'])
        
        # Convert the csv_files array back to a set
        initialState['csv_files'] = set(initialState['csv_files'])

<<<<<<< HEAD
        # Execute the graph with the initial state
        result = test_graph(initialState)
=======
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
>>>>>>> 70e1b2a288c5fa460b8e61263608bc5032ec3565

        return jsonify({
            'status': 'success',
            'message': 'Files processed successfully',
            'result': result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
if __name__ == '__main__':
    # app.run(debug=True, port=8001)
    # Use the PORT environment variable or default to 8001
    # port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=8000)

# if __name__ == '__main__':
#     app.run(debug=True, port=8001) 