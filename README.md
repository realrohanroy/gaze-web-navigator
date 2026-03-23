# Gaze-Based Web Navigator 👁️🖱️

A **webcam-based eye-tracking system** that enables **hands-free web navigation** using gaze input.  
The system estimates eye gaze in real time, maps it to screen coordinates through **user-specific calibration**, and controls a browser cursor via **WebSockets**.

This project focuses on **Human–Computer Interaction (HCI)**, **accessibility**, and **real-time computer vision**, using only a standard RGB webcam (no special hardware).

---

## 🚀 Features

- Real-time eye gaze estimation using a standard webcam  
- Iris-based gaze direction detection using **MediaPipe FaceMesh**  
- User-specific **9-point calibration**  
- **Resolution-independent** gaze mapping (normalized coordinates)  
- Exponential smoothing to reduce gaze jitter  
- WebSocket-based communication between Python backend and browser  
- Browser-based gaze cursor overlay  
- Modular, extensible architecture (ready for dwell-to-click, pause, recalibration)

---

## 🧠 System Overview

### High-Level Architecture

```
Webcam
  ↓
MediaPipe FaceMesh
  ↓
Iris & eye landmark extraction
  ↓
Normalized gaze vector computation
  ↓
Calibration (linear regression)
  ↓
Smoothed gaze coordinates
  ↓
WebSocket (Python → Browser)
  ↓
Browser cursor rendering
```

The **backend** handles computer vision, gaze estimation, calibration, and signal smoothing.  
The **frontend** handles UI rendering and browser interaction logic.

---

## 🛠️ Tech Stack

### Backend
- Python 3.10  
- OpenCV  
- MediaPipe FaceMesh  
- NumPy  
- WebSockets (asyncio)

### Frontend
- HTML / CSS  
- Vanilla JavaScript  
- WebSocket API

### Concepts Used
- Computer Vision  
- Human–Computer Interaction (HCI)  
- Signal smoothing (Exponential Moving Average)  
- Linear regression  
- Real-time systems  
- Client–server communication

---

## 📂 Project Structure

```
gaze-web-navigator/
│
├── backend/
│   ├── main.py                 # Final gaze → browser pipeline
│   ├── calibration_debug.py    # Calibration logic (one-time)
│   ├── websocket_server.py     # WebSocket utilities
│
├── frontend/
│   ├── index.html              # Browser UI
│   ├── styles.css              # Cursor styling
│   └── gaze.js                 # WebSocket + cursor logic
│
├── data/
│   └── calibration.json        # Saved user calibration
│
├── README.md
└── demo.mp4
```

---

## ⚙️ How It Works

### 1. Eye & Iris Tracking
- MediaPipe FaceMesh detects facial landmarks  
- Iris centers and eye corner landmarks are extracted  
- Iris displacement relative to eye center gives gaze direction  

### 2. Normalized Gaze Vector
Raw pixel offsets are normalized by eye width, making gaze estimation:
- Independent of face size  
- Less sensitive to camera distance  

### 3. Calibration (Critical Step)
A **9-point calibration** is performed:
- User looks at known screen positions  
- Corresponding gaze values are recorded  
- Linear regression learns the mapping:

```
screen_x = a * gaze_x + b
screen_y = c * gaze_y + d
```

Calibration data is saved to disk (`calibration.json`) and reused across sessions.

### 4. Smoothing
Raw gaze data is noisy due to:
- Micro-saccades  
- Camera noise  
- Landmark jitter  

Exponential smoothing is applied:

```
smoothed = α * new + (1 − α) * previous
```

This significantly improves cursor stability while maintaining low latency.

### 5. Browser Integration
- Backend sends **normalized gaze coordinates (0–1)** over WebSockets  
- Browser scales them to the current viewport size  
- Cursor works correctly in fullscreen, windowed, or resized states  

---

## ▶️ Running the Project

### 1. Install dependencies
```bash
pip install opencv-python mediapipe numpy websockets
```

### 2. Run calibration (one time per setup)
```bash
python backend/calibration_debug.py
```
This generates:
```
data/calibration.json
```

### 3. Start backend
```bash
python backend/main.py
```

### 4. Start frontend
```bash
cd frontend
python -m http.server 5500
```

### 5. Open browser
```
http://localhost:5500
```

---

## ⚠️ Known Limitations

- Requires relatively stable head position after calibration  
- Accuracy degrades with large head movement  
- Glasses and poor lighting can affect tracking quality  
- Webcam-based tracking is less precise than IR eye trackers  

These limitations are expected for RGB webcam eye tracking and are documented by design.

---

## ♿ Accessibility Use Cases

This system can be extended for:
- Hands-free web browsing  
- Assistive technology for motor impairments  
- Eye-based UI navigation  
- Research in attention tracking and UX  

---

## 🧪 Future Improvements

- Dwell-to-click interaction  
- Pause / resume tracking hotkey  
- Head pose compensation  
- Dynamic re-calibration  
- Attention heatmaps  
- Multi-monitor support  

---

## 🎥 Demo

A short demo video (`demo.mp4`) demonstrates:
- Calibration process  
- Live gaze-controlled cursor  
- Browser interaction  

---


