"""
Flask Backend for Web-Based HTR Note-Taking
============================================

REST API Endpoints:
- GET  /api/notes                    - Get all notes
- POST /api/notes                    - Create new note
- GET  /api/notes/<id>               - Get single note
- PUT  /api/notes/<id>               - Update note
- DELETE /api/notes/<id>             - Delete note
- POST /api/air-writing/start        - Start air writing session
- POST /api/air-writing/stop         - Stop air writing session
- POST /api/air-writing/recognize    - Recognize & append to active note
- GET  /api/air-writing/poll         - Poll for new recognitions
- GET  /api/stats                    - Get statistics
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
import base64
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from notes_storage import NotesStorage
from htr_model import Seq2SeqAttention, LabelEncoder, DEFAULT_VOCAB, load_model

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Paths - relative to this file

MODEL_PATH = Path(r"D:/code/python/TEst/app/model/ema_model.pth")
NOTES_PATH = Path(r"D:/code/python/TEst/output/notes.json")

# Create output directory if needed
NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

# Model config (must match training)
IMG_HEIGHT = 64
MAX_WIDTH = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*70}")
print("FLASK HTR NOTE-TAKING SERVER")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")
print(f"Storage: {NOTES_PATH}")

# Initialize storage
notes_storage = NotesStorage(NOTES_PATH)

# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

if MODEL_PATH.exists():
    try:
        model, encoder = load_model(MODEL_PATH, device=DEVICE, vocab=DEFAULT_VOCAB)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        encoder = LabelEncoder()
        encoder.build_vocab(DEFAULT_VOCAB)
        class DummyModel:
            def predict(self, image_tensor, encoder):
                return "test", 0.5
        model = DummyModel()
else:
    print(f"⚠ Model not found at: {MODEL_PATH}")
    print("  Using dummy model for testing")
    encoder = LabelEncoder()
    encoder.build_vocab(DEFAULT_VOCAB)
    class DummyModel:
        def predict(self, image_tensor, encoder):
            return "test", 0.5
    model = DummyModel()

# ══════════════════════════════════════════════════════════════════════════
# AIR WRITING SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

air_writing_session = {
    'active': False,
    'note_id': None,
    'recognitions': [],
    'recognition_counter': 0
}

@app.route('/api/air-writing/start', methods=['POST'])
def start_air_writing():
    """Start air writing session for a specific note"""
    try:
        data = request.get_json()
        note_id = data.get('note_id')
        
        if not note_id:
            return jsonify({'success': False, 'error': 'No note_id provided'}), 400
        
        air_writing_session['active'] = True
        air_writing_session['note_id'] = note_id
        air_writing_session['recognitions'] = []
        
        print(f"\n✓ Air writing session started for note #{note_id}")
        print("  Run: python air_writer.py")
        
        return jsonify({'success': True, 'note_id': note_id})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/air-writing/stop', methods=['POST'])
def stop_air_writing():
    """Stop air writing session"""
    try:
        air_writing_session['active'] = False
        air_writing_session['note_id'] = None
        
        print("\n✓ Air writing session stopped")
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/air-writing/poll', methods=['GET'])
def poll_air_writing():
    """Poll for new recognitions"""
    try:
        last_id = int(request.args.get('last_id', 0))
        
        new_recognitions = [
            r for r in air_writing_session['recognitions']
            if r['id'] > last_id
        ]
        
        return jsonify({
            'success': True,
            'recognitions': new_recognitions,
            'active': air_writing_session['active']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/air-writing/recognize', methods=['POST'])
def air_writing_recognize():
    """Recognition endpoint for air writing - appends to active note"""
    try:
        if not air_writing_session['active']:
            return jsonify({
                'success': False, 
                'error': 'No active air writing session. Click Start Air Writing in browser first.'
            }), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        print("\n" + "="*70)
        print("AIR WRITING RECOGNITION")
        print("="*70)
        print(f"Active note: #{air_writing_session['note_id']}")
        
        # Preprocess
        print("Preprocessing...")
        img_tensor = preprocess_canvas_image(data['image'])
        
        # Run model
        print("Running model...")
        predicted_text, confidence = model.predict(img_tensor, encoder)
        
        print(f"✓ Predicted: '{predicted_text}'")
        print(f"  Confidence: {confidence:.2%}")
        
        # Store recognition
        air_writing_session['recognition_counter'] += 1
        recognition = {
            'id': air_writing_session['recognition_counter'],
            'text': predicted_text,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        air_writing_session['recognitions'].append(recognition)
        
        # Append to active note
        note_id = air_writing_session['note_id']
        note = notes_storage.get_note(note_id)
        
        if note:
            current_text = note['text']
            new_text = current_text + (' ' if current_text else '') + predicted_text
            notes_storage.update_note(note_id, new_text)
            print(f"✓ Appended to note #{note_id}")
        
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'text': predicted_text,
            'confidence': float(confidence),
            'recognition_id': recognition['id']
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════
# HTR PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def preprocess_canvas_image(base64_data: str) -> torch.Tensor:
    """
    Preprocess canvas image - EXACT MATCH to working app_enhanced.py
    """
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        print(f"  Original canvas: {img.shape}")
        print(f"  Mean pixel value: {img.mean():.1f}")
        
        # Find white pixels (text) directly - NO INVERSION YET
        pts = cv2.findNonZero(img)
        
        if pts is None:
            raise ValueError("Canvas is empty")
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(pts)
        print(f"  Text bounding box: x={x}, y={y}, w={bw}, h={bh}")
        
        # Add padding
        pad = 14
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + bw + pad)
        y2 = min(img.shape[0], y + bh + pad)
        
        # Crop to content
        crop = img[y1:y2, x1:x2]
        print(f"  After crop: {crop.shape}")
        
        # INVERT: white text on black → black text on white
        # Use bitwise_not instead of 255 - img
        crop = cv2.bitwise_not(crop)
        print(f"  Mean after invert: {crop.mean():.1f}")
        
        # Resize maintaining aspect ratio
        h, w = crop.shape
        tw = int(w * (IMG_HEIGHT / h))
        tw = max(32, min(tw, MAX_WIDTH))
        img_resized = cv2.resize(crop, (tw, IMG_HEIGHT))
        
        print(f"  Final: {IMG_HEIGHT}x{tw}")
        
        # Normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        print(f"  Pixel range: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
        print(f"  Mean: {img_norm.mean():.3f}")
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img_norm).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(DEVICE)
    
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        raise
# ══════════════════════════════════════════════════════════════════════════
# ROUTES - FRONTEND
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve main web interface"""
    return render_template('index.html')

# ══════════════════════════════════════════════════════════════════════════
# ROUTES - NOTES CRUD
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/notes', methods=['GET'])
def get_notes():
    """Get all notes"""
    try:
        notes = notes_storage.get_all_notes()
        return jsonify({'success': True, 'notes': notes})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notes', methods=['POST'])
def create_note():
    """Create a new note"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        note = notes_storage.create_note(
            text=data['text'],
            confidence=data.get('confidence', 0.0)
        )
        
        return jsonify({'success': True, 'note': note}), 201
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notes/<int:note_id>', methods=['GET'])
def get_note(note_id):
    """Get a single note"""
    try:
        note = notes_storage.get_note(note_id)
        if note is None:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        return jsonify({'success': True, 'note': note})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notes/<int:note_id>', methods=['PUT'])
def update_note(note_id):
    """Update a note"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        note = notes_storage.update_note(note_id, data['text'])
        if note is None:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        
        return jsonify({'success': True, 'note': note})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/notes/<int:note_id>', methods=['DELETE'])
def delete_note(note_id):
    """Delete a note"""
    try:
        deleted = notes_storage.delete_note(note_id)
        if not deleted:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        
        return jsonify({'success': True, 'message': 'Note deleted'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get storage statistics"""
    try:
        stats = notes_storage.get_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Close storage on shutdown"""
    notes_storage.close()

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("HTR NOTE-TAKING WEB APP")
    print(f"{'='*70}")
    print(f"URL: http://localhost:5000")
    print(f"API: http://localhost:5000/api/notes")
    print(f"{'='*70}\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )