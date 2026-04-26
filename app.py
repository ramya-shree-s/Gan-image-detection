from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import traceback
from utils.analyzer import analyze_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Allow cross-origin requests ──
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
        file.save(filepath)
        result = analyze_image(filepath)
        return jsonify(result)

    except Exception as e:
        print("ERROR during analysis:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  GAN DETECTOR SERVER STARTING")
    print("="*50)
    print("  Open this in your browser:")
    print("  → http://localhost:5000")
    print("  DO NOT open index.html directly!")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)