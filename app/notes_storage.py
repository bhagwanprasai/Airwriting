"""
Simple Notes Storage - Text Only, No Images
============================================

Stores only recognized text with timestamps.
CRUD operations: Create, Read, Update, Delete

notes.json schema:
[
  {
    "id": 1,
    "text": "hello world",
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:05:00",
    "confidence": 0.95
  },
  ...
]
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class NotesStorage:
    """Simple JSON-based note storage with CRUD operations."""
    
    def __init__(self, storage_path: str | Path = "output/notes.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._notes: List[Dict] = self._load_notes()
        print(f"Notes Storage: {self.storage_path.resolve()}")
        print(f"  Loaded {len(self._notes)} notes")
    
    def _load_notes(self) -> List[Dict]:
        """Load notes from JSON file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load notes: {e}")
                return []
        return []
    
    def _save_notes(self) -> None:
        """Save notes to JSON file (atomic write)."""
        temp_path = self.storage_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self._notes, f, indent=2, ensure_ascii=False)
        temp_path.replace(self.storage_path)
    
    @staticmethod
    def _now() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat(timespec='seconds')
    
    def _next_id(self) -> int:
        """Get next available ID."""
        return max((n['id'] for n in self._notes), default=0) + 1
    
    # ══════════════════════════════════════════════════════════════════════
    # CREATE
    # ══════════════════════════════════════════════════════════════════════
    
    def create_note(self, text: str, confidence: float = 0.0) -> Dict:
        """Create a new note and return it."""
        with self._lock:
            note = {
                'id': self._next_id(),
                'text': text,
                'created_at': self._now(),
                'updated_at': self._now(),
                'confidence': round(confidence, 4)
            }
            self._notes.append(note)
            self._save_notes()
            return note
    
    # ══════════════════════════════════════════════════════════════════════
    # READ
    # ══════════════════════════════════════════════════════════════════════
    
    def get_all_notes(self) -> List[Dict]:
        """Get all notes, newest first."""
        with self._lock:
            return sorted(self._notes, key=lambda x: x['created_at'], reverse=True)
    
    def get_note(self, note_id: int) -> Optional[Dict]:
        """Get a single note by ID."""
        with self._lock:
            for note in self._notes:
                if note['id'] == note_id:
                    return dict(note)
            return None
    
    def search_notes(self, query: str) -> List[Dict]:
        """Search notes by text content."""
        with self._lock:
            query_lower = query.lower()
            return [
                note for note in self._notes
                if query_lower in note['text'].lower()
            ]
    
    # ══════════════════════════════════════════════════════════════════════
    # UPDATE
    # ══════════════════════════════════════════════════════════════════════
    
    def update_note(self, note_id: int, new_text: str) -> Optional[Dict]:
        """Update note text and return updated note."""
        with self._lock:
            for note in self._notes:
                if note['id'] == note_id:
                    note['text'] = new_text
                    note['updated_at'] = self._now()
                    self._save_notes()
                    return dict(note)
            return None
    
    # ══════════════════════════════════════════════════════════════════════
    # DELETE
    # ══════════════════════════════════════════════════════════════════════
    
    def delete_note(self, note_id: int) -> bool:
        """Delete a note by ID. Returns True if deleted."""
        with self._lock:
            original_count = len(self._notes)
            self._notes = [n for n in self._notes if n['id'] != note_id]
            if len(self._notes) < original_count:
                self._save_notes()
                return True
            return False
    
    def delete_all_notes(self) -> int:
        """Delete all notes. Returns count of deleted notes."""
        with self._lock:
            count = len(self._notes)
            self._notes = []
            self._save_notes()
            return count
    
    # ══════════════════════════════════════════════════════════════════════
    # STATS
    # ══════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        with self._lock:
            return {
                'total_notes': len(self._notes),
                'total_characters': sum(len(n['text']) for n in self._notes),
                'avg_confidence': (
                    sum(n['confidence'] for n in self._notes) / len(self._notes)
                    if self._notes else 0.0
                )
            }
    
    # ══════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════
    
    def export_to_text(self, output_path: str | Path) -> None:
        """Export all notes to a plain text file."""
        with self._lock:
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                for note in sorted(self._notes, key=lambda x: x['created_at']):
                    f.write(f"[{note['created_at']}] {note['text']}\n")
            print(f"✓ Exported {len(self._notes)} notes to {output_path}")
    
    # ══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════
    
    def close(self) -> None:
        """Ensure notes are saved."""
        with self._lock:
            self._save_notes()
        print("✓ Notes storage closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test CRUD operations
    storage = NotesStorage("test_notes.json")
    
    # CREATE
    note1 = storage.create_note("hello world", confidence=0.95)
    note2 = storage.create_note("test note", confidence=0.88)
    print(f"Created: {note1}")
    print(f"Created: {note2}")
    
    # READ
    all_notes = storage.get_all_notes()
    print(f"\nAll notes ({len(all_notes)}):")
    for note in all_notes:
        print(f"  {note['id']}: {note['text']}")
    
    # UPDATE
    updated = storage.update_note(note1['id'], "hello updated world")
    print(f"\nUpdated: {updated}")
    
    # SEARCH
    results = storage.search_notes("hello")
    print(f"\nSearch 'hello': {len(results)} results")
    
    # DELETE
    deleted = storage.delete_note(note2['id'])
    print(f"\nDeleted note {note2['id']}: {deleted}")
    
    # STATS
    stats = storage.get_stats()
    print(f"\nStats: {stats}")
    
    # EXPORT
    storage.export_to_text("exported_notes.txt")
    
    storage.close()