"""
Air Writer with HTR Model Integration - FastAPI Compatible
===========================================================
Write in the air with hand gestures → Real HTR recognition

Gestures:
  Index finger only (middle down)  →  Draw
  Index + middle fingers up        →  Hold / pen lifted
  
  R  →  Recognize with HTR model & append to web app note
  C  →  Clear canvas
  S  →  Save canvas PNG locally
  Q / Esc  →  Quit

Requirements:
  pip install mediapipe==0.10.14 opencv-python numpy requests torch
"""

import cv2
import numpy as np
import requests
import torch
import base64
from pathlib import Path
import sys

try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands

# Import HTR model
sys.path.append(str(Path(__file__).parent))
from htr_model import Seq2SeqAttention, LabelEncoder, DEFAULT_VOCAB, load_model

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Web app API endpoint (FastAPI runs on port 8000)
WEB_APP_URL = "http://localhost:8000"
API_AIR_WRITING = f"{WEB_APP_URL}/api/air-writing/recognize"

# Hardcoded paths
MODEL_PATH = Path(r"D:/code/python/TEst/app/model/ema_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Canvas size
CANVAS_W, CANVAS_H = 640, 480

# Drawing settings
PEN_COLOR = (255, 255, 255)
PEN_THICK = 5

# Smoothing
ALPHA_CURSOR = 0.25
ALPHA_STROKE = 0.45

# Gesture detection
DEBOUNCE_FRAMES = 5
GRACE_FRAMES = 10
MIN_MOVE_PX = 1

# ══════════════════════════════════════════════════════════════════════════
# EMA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential moving average"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None

# ══════════════════════════════════════════════════════════════════════════
# GESTURE STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════

class GestureState:
    def __init__(self):
        self.drawing = False
        self.index_up_counter = 0
        self.both_up_counter = 0
        self.grace_counter = 0
        
        self.cursor_ema = [EMA(ALPHA_CURSOR), EMA(ALPHA_CURSOR)]
        self.stroke_ema = [EMA(ALPHA_STROKE), EMA(ALPHA_STROKE)]
        
        self.last_draw_pos = None
    
    def update(self, hand_landmarks):
        """Process hand landmarks and update gesture state"""
        if hand_landmarks is None:
            self.grace_counter += 1
            if self.grace_counter > GRACE_FRAMES:
                self.reset()
            return None, None
        
        self.grace_counter = 0
        
        # Get finger tips
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        
        # Check if fingers extended
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        
        # Update counters
        if index_up and not middle_up:
            self.index_up_counter += 1
            self.both_up_counter = 0
        elif index_up and middle_up:
            self.both_up_counter += 1
            self.index_up_counter = 0
        else:
            self.index_up_counter = 0
            self.both_up_counter = 0
        
        # State transitions
        if self.index_up_counter >= DEBOUNCE_FRAMES:
            self.drawing = True
        elif self.both_up_counter >= DEBOUNCE_FRAMES:
            self.drawing = False
        
        # Get cursor position
        cursor_x = int(index_tip.x * CANVAS_W)
        cursor_y = int(index_tip.y * CANVAS_H)
        cursor_x = self.cursor_ema[0].update(cursor_x)
        cursor_y = self.cursor_ema[1].update(cursor_y)
        
        # Get stroke position
        if self.drawing:
            stroke_x = self.stroke_ema[0].update(cursor_x)
            stroke_y = self.stroke_ema[1].update(cursor_y)
            return (int(cursor_x), int(cursor_y)), (int(stroke_x), int(stroke_y))
        else:
            self.stroke_ema[0].reset()
            self.stroke_ema[1].reset()
            self.last_draw_pos = None
            return (int(cursor_x), int(cursor_y)), None
    
    def reset(self):
        self.drawing = False
        self.index_up_counter = 0
        self.both_up_counter = 0
        self.grace_counter = 0
        self.cursor_ema[0].reset()
        self.cursor_ema[1].reset()
        self.stroke_ema[0].reset()
        self.stroke_ema[1].reset()
        self.last_draw_pos = None

# ══════════════════════════════════════════════════════════════════════════
# WEB APP CLIENT
# ══════════════════════════════════════════════════════════════════════════

class WebAppClient:
    """Client to communicate with FastAPI web app"""
    
    @staticmethod
    def recognize_and_send(canvas):
        """
        Send RAW canvas to API.
        FastAPI handles ALL preprocessing (invert, crop, resize).
        """
        try:
            # Convert canvas to grayscale if needed
            if len(canvas.shape) == 3:
                img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = canvas
            
            # Encode RAW canvas to base64 (NO preprocessing here!)
            _, buffer = cv2.imencode('.png', img_gray)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_data = f"data:image/png;base64,{img_base64}"
            
            # Send to API
            response = requests.post(
                API_AIR_WRITING,
                json={'image': img_data},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return True, result.get('text'), result.get('confidence')
            
            return False, None, 0
        
        except Exception as e:
            print(f"API error: {e}")
            return False, None, 0
    
    @staticmethod
    def check_connection():
        """Check if web app is running"""
        try:
            response = requests.get(f"{WEB_APP_URL}/api/stats", timeout=2)
            return response.status_code == 200
        except:
            return False

# ══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("AIR WRITER - HTR MODEL INTEGRATION (FastAPI)")
    print("="*70)
    print("Gestures:")
    print("  Index finger only → Draw")
    print("  Both fingers up   → Hold (pen lifted)")
    print("\nControls:")
    print("  R → Recognize & append to web app note")
    print("  C → Clear canvas")
    print("  S → Save PNG locally")
    print("  Q → Quit")
    print("="*70)
    
    # Check web app
    print("\nChecking web app connection...")
    if WebAppClient.check_connection():
        print(f"✓ Connected to {WEB_APP_URL}")
    else:
        print(f"⚠ Cannot connect to {WEB_APP_URL}")
        print("  Make sure FastAPI is running:")
        print("  python fastapi_app.py")
        print("  OR: uvicorn fastapi_app:app --reload")
        print("  And you clicked 'Start Air Writing' in browser")
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        input("Press Enter to exit...")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✓ Webcam ready")
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize state
    gesture_state = GestureState()
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    
    print("\n" + "="*70)
    print("✓ READY! Show your hand to the camera")
    print("="*70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontally
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Update gesture
        hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
        cursor_pos, stroke_pos = gesture_state.update(hand_landmarks)
        
        # Draw on canvas
        if stroke_pos and gesture_state.drawing:
            if gesture_state.last_draw_pos:
                dx = stroke_pos[0] - gesture_state.last_draw_pos[0]
                dy = stroke_pos[1] - gesture_state.last_draw_pos[1]
                if abs(dx) >= MIN_MOVE_PX or abs(dy) >= MIN_MOVE_PX:
                    cv2.line(canvas, gesture_state.last_draw_pos, stroke_pos, PEN_COLOR, PEN_THICK)
                    gesture_state.last_draw_pos = stroke_pos
            else:
                gesture_state.last_draw_pos = stroke_pos
        elif not gesture_state.drawing:
            gesture_state.last_draw_pos = None
        
        # Draw cursor
        if cursor_pos:
            color = (0, 255, 0) if gesture_state.drawing else (0, 0, 255)
            cv2.circle(frame, cursor_pos, 10, color, -1)
        
        # Draw hand skeleton
        if hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
        
        # Add status
        status = "DRAWING" if gesture_state.drawing else "READY"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if gesture_state.drawing else (0, 0, 255), 2)
        cv2.putText(frame, "R:Recognize C:Clear S:Save Q:Quit", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show windows
        cv2.imshow('Air Writer - Camera', frame)
        cv2.imshow('Air Writer - Canvas', canvas)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        
        elif key == ord('c'):
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            print("✓ Canvas cleared")
        
        elif key == ord('s'):
            filename = f"airwrite_{cv2.getTickCount()}.png"
            cv2.imwrite(filename, canvas)
            print(f"✓ Saved: {filename}")
        
        elif key == ord('r'):
            print("\n" + "─"*50)
            print("Recognizing...")
            
            # Check if canvas has content
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if len(canvas.shape) == 3 else canvas
            if canvas_gray.max() == 0:
                print("⚠ Canvas is empty")
                print("─"*50 + "\n")
                continue
            
            try:
                # Send RAW canvas to FastAPI (FastAPI does ALL preprocessing)
                success, text, confidence = WebAppClient.recognize_and_send(canvas)
                
                if success:
                    print(f"✓ Recognized: '{text}'")
                    print(f"  Confidence: {confidence:.2%}")
                    print(f"✓ Appended to web app note")
                    
                    # Clear canvas
                    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                    print(f"✓ Canvas cleared - ready for next word")
                else:
                    print("⚠ Failed - no active air writing session")
                    print("  Click 'Start Air Writing' in browser first")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print("─"*50 + "\n")
    
    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\n✓ Air Writer closed")

if __name__ == "__main__":
    main()