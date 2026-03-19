"""
Storage Manager — JSON + Image File Storage
============================================

Layout (all inside OUTPUT_DIR):
  output/
    predictions.json   ← saved predictions
    cache.json         ← prediction cache (canvas hash → result)
    airwrite_<ts>.png  ← one PNG per prediction

predictions.json schema:
  [
    {
      "id":         1,
      "text":       "hello",
      "image_file": "airwrite_20240101_120000_1.png",
      "confidence": 0.97,
      "saved_at":   "2024-01-01T12:00:00"
    },
    ...
  ]

cache.json schema:
  {
    "<canvas_md5>": {
      "prediction":    "hello",
      "confidence":    0.97,
      "model_version": "v1.0",
      "hit_count":     3,
      "created_at":    "2024-01-01T12:00:00"
    },
    ...
  }
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageManager:
    """
    Stores predicted text + canvas PNG images as plain files.
    No database, no setup — just a folder.
    Thread-safe via RLock.
    """

    PREDICTIONS_FILE = "predictions.json"
    CACHE_FILE       = "cache.json"

    def __init__(self, output_dir: str | Path = "output") -> None:
        self._dir        = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pred_path  = self._dir / self.PREDICTIONS_FILE
        self._cache_path = self._dir / self.CACHE_FILE
        self._lock       = threading.RLock()

        self._predictions: List[Dict]       = self._load_json(self._pred_path, [])
        self._cache:       Dict[str, Dict]  = self._load_json(self._cache_path, {})

        print(f"✓ Storage: {self._dir.resolve()}")
        print(f"  {len(self._predictions)} predictions  |  {len(self._cache)} cache entries")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠  Could not read {path.name}: {e} — starting fresh")
        return default

    def _write_json(self, path: Path, data) -> None:
        """Atomic write: temp file → rename, so a crash never corrupts the file."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _next_id(self) -> int:
        return max((p["id"] for p in self._predictions), default=0) + 1

    # ── Predictions ───────────────────────────────────────────────────────────

    def save_note(
        self,
        title: str,
        image_data: Optional[bytes] = None,
        confidence: float = 0.0,
        # legacy kwargs accepted but ignored
        **_kwargs,
    ) -> int:
        """
        Save predicted text and its canvas image.
        Returns the new record id.
        """
        with self._lock:
            new_id   = self._next_id()
            img_file = None

            if image_data:
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_file = f"airwrite_{ts}_{new_id}.png"
                try:
                    (self._dir / img_file).write_bytes(image_data)
                except Exception as e:
                    print(f"  ⚠ Could not save image: {e}")
                    img_file = None

            record = {
                "id":         new_id,
                "text":       title,          # predicted word / phrase
                "image_file": img_file,       # filename only; None if encoding failed
                "confidence": round(confidence, 4),
                "saved_at":   self._now(),
            }

            self._predictions.append(record)
            self._write_json(self._pred_path, self._predictions)
            return new_id

    def get_notes_count(self) -> int:
        with self._lock:
            return len(self._predictions)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def get_cached_prediction(self, canvas_hash: str) -> Optional[Dict]:
        with self._lock:
            entry = self._cache.get(canvas_hash)
            if entry:
                entry["hit_count"] = entry.get("hit_count", 1) + 1
                self._write_json(self._cache_path, self._cache)
                return dict(entry)
            return None

    def save_cached_prediction(
        self,
        canvas_hash: str,
        prediction: str,
        confidence: float = 0.0,
        model_version: str = "v1",
    ) -> bool:
        with self._lock:
            existing = self._cache.get(canvas_hash)
            if existing:
                existing.update({
                    "prediction":    prediction,
                    "confidence":    round(confidence, 4),
                    "model_version": model_version,
                    "hit_count":     existing.get("hit_count", 1) + 1,
                })
            else:
                self._cache[canvas_hash] = {
                    "prediction":    prediction,
                    "confidence":    round(confidence, 4),
                    "model_version": model_version,
                    "hit_count":     1,
                    "created_at":    self._now(),
                }
            self._write_json(self._cache_path, self._cache)
            return True

    def clear_old_cache(self, days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock:
            stale = [k for k, v in self._cache.items()
                     if v.get("created_at", "9999") < cutoff]
            for k in stale:
                del self._cache[k]
            if stale:
                self._write_json(self._cache_path, self._cache)
            return len(stale)

    def get_cache_stats(self) -> Dict:
        with self._lock:
            return {
                "count":      len(self._cache),
                "total_hits": sum(v.get("hit_count", 0) for v in self._cache.values()),
            }

    # ── Info / lifecycle ──────────────────────────────────────────────────────

    def get_database_info(self) -> Dict[str, Any]:
        return {
            "type":      "json",
            "connected": True,
            "path":      str(self._dir.resolve()),
        }

    def close(self) -> None:
        print("✓ Storage closed (files are up to date)")

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()