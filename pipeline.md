# Air Writing OCR Pipeline Architecture

This document outlines the end-to-end data flow for the Air Writing application. The system uses computer vision to track hand gestures, captures "air ink," processes the visual data, and converts it into digital text using a CRNN model.

## 1. System Overview

* **Input:** Webcam Video Feed (Real-time).
* **Interaction:** "Pinch" gesture (Thumb + Index finger) to simulate a pen down event.
* **Trigger:** 5-second inactivity timer (auto-send).
* **Core Engine:** MediaPipe (New Tasks API) + PyTorch (CRNN).
* **Output:** Saved Image (`.png`) + Saved Text (`.txt`).

---

## 2. Pipeline Stages

### Stage 1: Vision & Tracking (The "Eye")
* **Source:** OpenCV VideoCapture (30 FPS).
* **Model:** MediaPipe `HandLandmarker` (New Tasks API).
* **Operation:**
    1.  Flip frame horizontally (Mirror view).
    2.  Convert BGR to RGB.
    3.  Extract landmarks for **Index Tip (8)**, **Thumb Tip (4)**, and **Wrist (0)**.

### Stage 2: Gesture Logic (The "Brain")
* **Pinch Detection:**
    * Calculate Euclidean distance between *Index Tip* and *Thumb Tip*.
    * **Normalization:** Divide pinch distance by *Hand Scale* (Wrist to Middle Knuckle distance) to ensure consistency at different camera depths.
    * **Hysteresis:** Use two thresholds (Start vs. Stop) to prevent flickering.
* **Cursor Smoothing:**
    * Apply Exponential Moving Average (EMA) to the coordinate stream to remove webcam jitter.
    * Target Point: Midpoint between Thumb and Index finger.

### Stage 3: Canvas Management (The "Paper")
* **State:**
    * **Drawing (Pinch Active):** Draw thick lines (Cyan/White) on a transparent overlay layer. Reset the Inactivity Timer.
    * **Hovering (Pinch Inactive):** Move cursor without drawing.
* **Inactivity Trigger:**
    * A background timer runs continuously.
    * If `Time Since Last Draw > 5.0 Seconds`:
        1.  Lock the canvas.
        2.  Trigger the **Inference Pipeline**.

### Stage 4: Preprocessing (The "Lens")
Before the AI sees the image, it must be cleaned to match the training data (IAM Dataset format).
1.  **Extraction:** Crop the canvas to the bounding box of the drawing (remove empty space).
2.  **Padding:** Add black borders to center the text (prevents "Zoom Effect").
3.  **Binarization:** Apply Thresholding to ensure high contrast (White Text / Black Background).
4.  **Resizing:** Resize height to **64px**, keeping aspect ratio.
5.  **Normalization:** Scale pixel values from `[0, 255]` to `[0.0, 1.0]`.

### Stage 5: Inference (The "Translator")
* **Model:** Deep CRNN (Convolutional Recurrent Neural Network).
    * *CNN Layers:* Extract visual features.
    * *Bi-LSTM Layers:* Analyze sequence context.
* **Decoder:**
    * CTC (Connectionist Temporal Classification) Greedy Decoder.
    * Converts probability matrix -> Character indices -> Final String.

### Stage 6: Output & Storage (The "Archivist")
1.  **File System Operations:**
    * **Image:** Save the preprocessed crop as `scans/word_{timestamp}.png`.
    * **Text:** Append the recognized string to `notes.txt`.
2.  **UI Feedback:**
    * Display recognized text on screen.
    * Clear the canvas for the next word.

---

## 3. Data Flow Diagram

```mermaid
graph TD
    A[Webcam Feed] --> B(MediaPipe HandLandmarker)
    B --> C{Pinch Detected?}
    C -- Yes --> D[Draw Line on Canvas]
    C -- No --> E[Update Cursor / Hover]
    
    D --> F[Reset Timer]
    E --> G{Timer > 5.0s?}
    
    G -- No --> A
    G -- Yes --> H[Trigger Processing]
    
    H --> I[Crop & Pad Image]
    I --> J[Preprocess: Binarize & Resize to 64px]
    
    J --> K[[PyTorch CRNN Model]]
    K --> L[Decode Text]
    
    L --> M[Save: notes.txt]
    J --> N[Save: scans/word_timestamp.png]
    
    M --> O[Clear Canvas]
    N --> O