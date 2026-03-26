"""
FastAPI Backend for Web-Based HTR Note-Taking - IMPROVED VERSION
=================================================================

Improvements:
- Preprocessing pipeline matches training exactly
- Better error handling and stability
- Improved WebSocket management
- Session state validation
- Comprehensive logging

REST API Endpoints:
- GET    /api/notes                    - Get all notes
- POST   /api/notes                    - Create new note
- GET    /api/notes/{id}               - Get single note
- PUT    /api/notes/{id}               - Update note
- DELETE /api/notes/{id}               - Delete note
- POST   /api/air-writing/start        - Start air writing session
- POST   /api/air-writing/stop         - Stop air writing session
- POST   /api/air-writing/recognize    - Recognize & append to active note
- GET    /api/air-writing/poll         - Poll for new recognitions
- GET    /api/stats                    - Get statistics
- WS     /ws/notes                     - WebSocket for real-time note updates
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import torch
import numpy as np
import cv2
import base64
from pathlib import Path
import sys
import asyncio
import json
import logging
import traceback
from contextlib import contextmanager
import threading
import sqlite3

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from htr_model import Seq2SeqAttention, LabelEncoder, DEFAULT_VOCAB, load_model

# ══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Paths - Use relative paths for portability
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "ema_model.pth"
DB_PATH = BASE_DIR / "data" / "notes.db"

# Create directories
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Model config - MUST MATCH TRAINING (from model_monolithic.py Config)
IMG_HEIGHT = 64
MIN_WIDTH = 32
MAX_WIDTH = 512
ENCODER_HIDDEN = 256
DECODER_HIDDEN = 256
ATTENTION_HIDDEN = 128
DROPOUT = 0.3
MAX_OUTPUT_LENGTH = 32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"{'='*70}")
logger.info("FASTAPI HTR NOTE-TAKING SERVER - IMPROVED VERSION")
logger.info(f"{'='*70}")
logger.info(f"Device: {DEVICE}")
logger.info(f"Model: {MODEL_PATH}")
logger.info(f"Database: {DB_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (Request/Response Validation)
# ══════════════════════════════════════════════════════════════════════════

class NoteBase(BaseModel):
    text: str = Field(..., description="Note text content")
    
class NoteCreate(NoteBase):
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Recognition confidence")

class NoteUpdate(BaseModel):
    text: str = Field(..., min_length=0, description="Updated note text")

class NoteResponse(NoteBase):
    id: int = Field(..., description="Unique note ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    confidence: float = Field(..., description="Recognition confidence score")
    
    class Config:
        from_attributes = True

class NotesListResponse(BaseModel):
    success: bool = True
    notes: List[NoteResponse]

class NoteDetailResponse(BaseModel):
    success: bool = True
    note: NoteResponse

class DeleteResponse(BaseModel):
    success: bool = True
    message: str = "Note deleted"

class StatsResponse(BaseModel):
    success: bool = True
    stats: dict

class AirWritingStartRequest(BaseModel):
    note_id: int = Field(..., gt=0, description="ID of note to append recognized text")

class AirWritingStartResponse(BaseModel):
    success: bool = True
    note_id: int

class AirWritingStopResponse(BaseModel):
    success: bool = True

class AirWritingRecognizeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded canvas image")

class AirWritingRecognizeResponse(BaseModel):
    success: bool = True
    text: str = Field(..., description="Recognized text")
    confidence: float = Field(..., description="Recognition confidence")
    recognition_id: int = Field(..., description="Recognition ID for tracking")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════
# SQLITE DATABASE
# ══════════════════════════════════════════════════════════════════════════

class NotesDatabase:
    """SQLite-based note storage with connection pooling and thread safety"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"Database initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Thread-safe connection context manager"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            logger.error(f"Database error: {e}")
            if self._local.connection:
                self._local.connection.rollback()
            raise
    
    def _init_db(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notes_created_at 
                ON notes(created_at DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notes_updated_at 
                ON notes(updated_at DESC)
            """)
            
            conn.commit()
    
    async def create_note(self, text: str, confidence: float = 0.0) -> dict:
        """Create a new note"""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO notes (text, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (text, confidence, datetime.now(), datetime.now())
                )
                conn.commit()
                
                note_id = cursor.lastrowid
                return await self.get_note(note_id)
    
    async def get_note(self, note_id: int) -> Optional[dict]:
        """Get a single note by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM notes WHERE id = ?",
                (note_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    async def get_all_notes(self) -> List[dict]:
        """Get all notes, newest first"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM notes ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    async def update_note(self, note_id: int, text: str) -> Optional[dict]:
        """Update note text"""
        with self._lock:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE notes 
                    SET text = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (text, datetime.now(), note_id)
                )
                conn.commit()
                
                if conn.total_changes > 0:
                    return await self.get_note(note_id)
                return None
    
    async def delete_note(self, note_id: int) -> bool:
        """Delete a note"""
        with self._lock:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
                conn.commit()
                return conn.total_changes > 0
    
    async def get_stats(self) -> dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_notes,
                    SUM(LENGTH(text)) as total_characters,
                    AVG(confidence) as avg_confidence
                FROM notes
            """)
            row = cursor.fetchone()
            
            return {
                'total_notes': row['total_notes'] or 0,
                'total_characters': row['total_characters'] or 0,
                'avg_confidence': row['avg_confidence'] or 0.0
            }
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

# ══════════════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER
# ══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        dead_connections = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for connection in dead_connections:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

# ══════════════════════════════════════════════════════════════════════════
# HTR PREPROCESSING - MATCHES TRAINING EXACTLY
# ══════════════════════════════════════════════════════════════════════════

class HTRPreprocessor:
    """
    Preprocessing pipeline for HTR model inference.
    Matches training preprocessing from model_monolithic.py exactly.
    """
    
    def __init__(self, img_height: int = 64, min_width: int = 32, max_width: int = 512):
        self.img_height = img_height
        self.min_width = min_width
        self.max_width = max_width
        self.debug = True  # Set to False in production
    
    def _log(self, message: str):
        """Debug logging"""
        if self.debug:
            logger.info(f"  {message}")
    
    def decode_base64(self, base64_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode as grayscale
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError("Could not decode image - invalid format")
            
            return img
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    def find_content_bbox(self, img: np.ndarray, threshold: int = 10) -> tuple:
        """
        Find bounding box of content (white pixels on black background).
        
        Args:
            img: Grayscale image (white text on black background)
            threshold: Minimum pixel value to consider as content
            
        Returns:
            (x, y, w, h) bounding box, or None if no content found
        """
        # Find pixels above threshold
        binary = img > threshold
        points = cv2.findNonZero((binary * 255).astype(np.uint8))
        
        if points is None:
            return None
        
        return cv2.boundingRect(points)
    
    def crop_to_content(self, img: np.ndarray, padding: int = 14) -> np.ndarray:
        """
        Crop image to content bounding box with padding.
        
        Args:
            img: Grayscale image
            padding: Padding around content in pixels
            
        Returns:
            Cropped image
        """
        bbox = self.find_content_bbox(img)
        
        if bbox is None:
            raise ValueError("Canvas is empty - no content detected")
        
        x, y, w, h = bbox
        self._log(f"Content bbox: x={x}, y={y}, w={w}, h={h}")
        
        # Add padding with boundary check
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        return img[y1:y2, x1:x2]
    
    def invert_colors(self, img: np.ndarray) -> np.ndarray:
        """
        Invert image colors.
        Training expects: BLACK TEXT on WHITE BACKGROUND
        Canvas has: WHITE TEXT on BLACK BACKGROUND
        
        Args:
            img: Input grayscale image
            
        Returns:
            Inverted image
        """
        return cv2.bitwise_not(img)
    
    def resize_with_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target height while maintaining aspect ratio.
        Matches training exactly from model_monolithic.py:
        
        scale = IMG_HEIGHT / h
        new_w = int(w * scale)
        new_w = max(MIN_WIDTH, min(new_w, MAX_WIDTH))
        image = cv2.resize(image, (new_w, IMG_HEIGHT))
        
        Args:
            img: Input grayscale image
            
        Returns:
            Resized image
        """
        h, w = img.shape
        
        # Calculate new width maintaining aspect ratio
        scale = self.img_height / h
        new_w = int(w * scale)
        
        # Clamp width to valid range
        new_w = max(self.min_width, min(new_w, self.max_width))
        
        self._log(f"Resize: {w}x{h} -> {new_w}x{self.img_height}")
        
        return cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_LINEAR)
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        Matches training: image.astype(np.float32) / 255.0
        
        Args:
            img: Input grayscale image (0-255)
            
        Returns:
            Normalized image (0-1)
        """
        return img.astype(np.float32) / 255.0
    
    def preprocess(self, base64_data: str) -> torch.Tensor:
        """
        Full preprocessing pipeline matching training exactly.
        
        Steps:
        1. Decode base64 to grayscale
        2. Crop to content (remove empty space)
        3. Invert colors (white-on-black -> black-on-white)
        4. Resize to target height maintaining aspect ratio
        5. Normalize to [0, 1]
        6. Convert to tensor [1, 1, H, W]
        
        Args:
            base64_data: Base64 encoded canvas image
            
        Returns:
            Preprocessed tensor ready for model
        """
        self._log("=" * 50)
        self._log("PREPROCESSING START")
        
        # Step 1: Decode
        img = self.decode_base64(base64_data)
        self._log(f"Original: {img.shape}, range=[{img.min()}, {img.max()}]")
        
        # Step 2: Crop to content
        try:
            img = self.crop_to_content(img, padding=14)
            self._log(f"After crop: {img.shape}")
        except ValueError as e:
            logger.warning(f"Content detection failed: {e}")
            raise
        
        # Step 3: Invert colors (white text on black -> black text on white)
        img = self.invert_colors(img)
        self._log(f"After invert: range=[{img.min()}, {img.max()}], mean={img.mean():.1f}")
        
        # Step 4: Resize with aspect ratio
        img = self.resize_with_aspect_ratio(img)
        self._log(f"After resize: {img.shape}")
        
        # Step 5: Normalize
        img = self.normalize(img)
        self._log(f"After normalize: range=[{img.min():.3f}, {img.max():.3f}], mean={img.mean():.3f}")
        
        # Step 6: Convert to tensor [1, 1, H, W]
        tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        
        self._log(f"Final tensor: {tensor.shape}")
        self._log("PREPROCESSING END")
        self._log("=" * 50)
        
        return tensor.to(DEVICE)
    
    def preprocess_with_debug_image(self, base64_data: str) -> tuple:
        """
        Preprocess and return both tensor and debug image.
        Useful for visualizing preprocessing steps.
        
        Returns:
            (tensor, debug_image_bytes)
        """
        import io
        
        img = self.decode_base64(base64_data)
        original = img.copy()
        
        img = self.crop_to_content(img, padding=14)
        cropped = img.copy()
        
        img = self.invert_colors(img)
        inverted = img.copy()
        
        img = self.resize_with_aspect_ratio(img)
        resized = img.copy()
        
        # Create debug visualization
        h, w = original.shape
        debug_img = np.ones((max(h, 128) + 10, w + 256 + 20), dtype=np.uint8) * 255
        
        # Original
        debug_img[10:10+h, 10:10+w] = original
        
        # Cropped (scaled to fit)
        ch, cw = cropped.shape
        scale = min(64 / ch, 128 / cw)
        scaled_crop = cv2.resize(cropped, (int(cw * scale), int(ch * scale)))
        debug_img[10:10+scaled_crop.shape[0], w+20:w+20+scaled_crop.shape[1]] = scaled_crop
        
        # Final
        fh, fw = resized.shape
        debug_img[10:10+fh, w+150:w+150+fw] = resized
        
        # Encode to PNG
        _, buffer = cv2.imencode('.png', debug_img)
        
        # Return tensor and debug image
        img_norm = self.normalize(resized)
        tensor = torch.FloatTensor(img_norm).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(DEVICE), buffer.tobytes()

# ══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="HTR Note-Taking API",
    description="Handwritten Text Recognition Note-Taking Application with Air Writing - Improved Version",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = NotesDatabase(DB_PATH)

# WebSocket manager
ws_manager = ConnectionManager()

# Preprocessor
preprocessor = HTRPreprocessor(
    img_height=IMG_HEIGHT,
    min_width=MIN_WIDTH,
    max_width=MAX_WIDTH
)

# Air writing session state with thread safety
class AirWritingSession:
    """Thread-safe air writing session management"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._note_id = None
        self._recognitions = []
        self._recognition_counter = 0
    
    @property
    def active(self) -> bool:
        with self._lock:
            return self._active
    
    @property
    def note_id(self) -> Optional[int]:
        with self._lock:
            return self._note_id
    
    def start(self, note_id: int):
        with self._lock:
            self._active = True
            self._note_id = note_id
            self._recognitions = []
            self._recognition_counter = 0
    
    def stop(self):
        with self._lock:
            self._active = False
            self._note_id = None
    
    def add_recognition(self, text: str, confidence: float) -> dict:
        with self._lock:
            self._recognition_counter += 1
            recognition = {
                'id': self._recognition_counter,
                'text': text,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            self._recognitions.append(recognition)
            return recognition
    
    def get_new_recognitions(self, last_id: int) -> List[dict]:
        with self._lock:
            return [r for r in self._recognitions if r['id'] > last_id]

air_writing_session = AirWritingSession()

# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

model = None
encoder = None

def init_model():
    """Initialize the HTR model"""
    global model, encoder
    
    if MODEL_PATH.exists():
        try:
            model, encoder = load_model(MODEL_PATH, device=DEVICE, vocab=DEFAULT_VOCAB)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            logger.warning("Using dummy model for testing")
            create_dummy_model()
    else:
        logger.warning(f"Model not found at: {MODEL_PATH}")
        logger.warning("Using dummy model for testing")
        create_dummy_model()

def create_dummy_model():
    """Create a dummy model for testing"""
    global model, encoder
    
    encoder = LabelEncoder()
    encoder.build_vocab(DEFAULT_VOCAB)
    
    class DummyModel:
        def predict(self, image_tensor, encoder):
            return "test", 0.5
    
    model = DummyModel()

# Initialize model on startup
init_model()

# ══════════════════════════════════════════════════════════════════════════
# API ROUTES - FRONTEND
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main web interface"""
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>HTR Notes App</h1><p>Place index.html in the same directory</p>")

# ══════════════════════════════════════════════════════════════════════════
# API ROUTES - NOTES CRUD
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/notes", response_model=NotesListResponse)
async def get_notes():
    """Get all notes"""
    try:
        notes = await db.get_all_notes()
        return NotesListResponse(notes=notes)
    except Exception as e:
        logger.error(f"Error getting notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notes", response_model=NoteDetailResponse, status_code=201)
async def create_note(note: NoteCreate, background_tasks: BackgroundTasks):
    """Create a new note"""
    try:
        created_note = await db.create_note(note.text, note.confidence)
        
        # Broadcast to WebSocket clients
        background_tasks.add_task(
            ws_manager.broadcast,
            {"type": "note_created", "note": created_note}
        )
        
        return NoteDetailResponse(note=created_note)
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notes/{note_id}", response_model=NoteDetailResponse)
async def get_note(note_id: int):
    """Get a single note"""
    note = await db.get_note(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteDetailResponse(note=note)

@app.put("/api/notes/{note_id}", response_model=NoteDetailResponse)
async def update_note(note_id: int, note_update: NoteUpdate, background_tasks: BackgroundTasks):
    """Update a note"""
    updated_note = await db.update_note(note_id, note_update.text)
    if updated_note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Broadcast to WebSocket clients
    background_tasks.add_task(
        ws_manager.broadcast,
        {"type": "note_updated", "note": updated_note}
    )
    
    return NoteDetailResponse(note=updated_note)

@app.delete("/api/notes/{note_id}", response_model=DeleteResponse)
async def delete_note(note_id: int, background_tasks: BackgroundTasks):
    """Delete a note"""
    deleted = await db.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Broadcast to WebSocket clients
    background_tasks.add_task(
        ws_manager.broadcast,
        {"type": "note_deleted", "note_id": note_id}
    )
    
    return DeleteResponse()

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get storage statistics"""
    stats = await db.get_stats()
    return StatsResponse(stats=stats)

# ══════════════════════════════════════════════════════════════════════════
# API ROUTES - AIR WRITING
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/air-writing/start", response_model=AirWritingStartResponse)
async def start_air_writing(request: AirWritingStartRequest):
    """Start air writing session for a specific note"""
    # Verify note exists
    note = await db.get_note(request.note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    
    air_writing_session.start(request.note_id)
    
    logger.info(f"Air writing session started for note #{request.note_id}")
    
    return AirWritingStartResponse(note_id=request.note_id)

@app.post("/api/air-writing/stop", response_model=AirWritingStopResponse)
async def stop_air_writing():
    """Stop air writing session"""
    air_writing_session.stop()
    logger.info("Air writing session stopped")
    return AirWritingStopResponse()

@app.get("/api/air-writing/poll")
async def poll_air_writing(last_id: int = 0):
    """Poll for new recognitions"""
    new_recognitions = air_writing_session.get_new_recognitions(last_id)
    
    return {
        'success': True,
        'recognitions': new_recognitions,
        'active': air_writing_session.active
    }

@app.post("/api/air-writing/recognize", response_model=AirWritingRecognizeResponse)
async def air_writing_recognize(
    request: AirWritingRecognizeRequest,
    background_tasks: BackgroundTasks
):
    """Recognition endpoint for air writing - appends to active note"""
    
    # Check if session is active
    if not air_writing_session.active:
        raise HTTPException(
            status_code=400,
            detail='No active air writing session. Click Start Air Writing in browser first.'
        )
    
    note_id = air_writing_session.note_id
    if note_id is None:
        raise HTTPException(status_code=400, detail='Invalid session state')
    
    logger.info("="*70)
    logger.info("AIR WRITING RECOGNITION")
    logger.info("="*70)
    logger.info(f"Active note: #{note_id}")
    
    try:
        # Preprocess image
        logger.info("Preprocessing...")
        img_tensor = preprocessor.preprocess(request.image)
        
        # Run model
        logger.info("Running model inference...")
        predicted_text, confidence = model.predict(img_tensor, encoder)
        
        logger.info(f"Predicted: '{predicted_text}'")
        logger.info(f"Confidence: {confidence:.2%}")
        
        # Store recognition
        recognition = air_writing_session.add_recognition(predicted_text, confidence)
        
        # Append to active note
        note = await db.get_note(note_id)
        
        if note:
            current_text = note['text']
            new_text = current_text + (' ' if current_text else '') + predicted_text
            updated_note = await db.update_note(note_id, new_text)
            logger.info(f"Appended to note #{note_id}")
            
            # Broadcast to WebSocket clients
            background_tasks.add_task(
                ws_manager.broadcast,
                {"type": "note_updated", "note": updated_note}
            )
        
        logger.info("="*70 + "\n")
        
        return AirWritingRecognizeResponse(
            text=predicted_text,
            confidence=float(confidence),
            recognition_id=recognition['id']
        )
    
    except ValueError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════
# WEBSOCKET - REAL-TIME UPDATES
# ══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/notes")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time note updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )
                # Echo back for heartbeat
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send ping to check connection
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)

# ══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }

# ══════════════════════════════════════════════════════════════════════════
# LIFECYCLE EVENTS
# ══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    logger.info(f"{'='*70}")
    logger.info("HTR NOTE-TAKING WEB APP - FASTAPI IMPROVED")
    logger.info(f"{'='*70}")
    logger.info(f"Server starting...")
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"Main App: http://localhost:8000")
    logger.info(f"{'='*70}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    db.close()
    logger.info("Database closed")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        "fastapi_app_improved:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )