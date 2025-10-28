# File: app/main.py

import sys
import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import socketio
import uvicorn
from collections import deque

# Thêm thư mục gốc vào Python Path
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

# Import các lớp xử lý AI
from src.inference import ActionRecognizer
from src.pose_extractor import MediaPipeExtractor # Vẫn cần để lấy xương

# --- KHỞI TẠO ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi')
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Đường dẫn model đã fine-tune
WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', 'best_finetuned_model.pt')

# Khởi tạo các đối tượng AI
recognizer = ActionRecognizer(WEIGHTS_PATH)
pose_extractor = MediaPipeExtractor()
skeleton_buffer = deque(maxlen=30)

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(PROJECT_ROOT_DIR, 'app', 'templates', 'index.html')
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

static_path = os.path.join(PROJECT_ROOT_DIR, 'app', 'static')
app.mount("/static", StaticFiles(directory=static_path), name="static")

# --- SOCKET.IO ---
@sio.on('connect')
def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.on('video_frame')
async def handle_video_frame(sid, data):
    print(f"--- Received frame from {sid} ---") # <-- Print 1

    frame = None # Khởi tạo frame là None
    try:
        # Tách phần data base64
        header, encoded = data.split(',', 1)
        # Decode base64
        image_data = base64.b64decode(encoded)
        # Chuyển thành mảng numpy
        np_arr = np.frombuffer(image_data, np.uint8)
        # Decode ảnh bằng OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("  -> Decoded frame successfully.") # <-- Print thêm

    except Exception as e:
        print(f"!!! Lỗi nghiêm trọng khi decode ảnh: {e}")
        # Không cần return, vì frame vẫn là None sẽ được xử lý bên dưới

    # Kiểm tra frame sau khi decode
    if frame is None:
        print("!!! Lỗi: Frame không hợp lệ sau khi decode.")
        # Có thể gửi lại lỗi cho client hoặc chỉ bỏ qua
        # Gửi lại trạng thái 'Đang chờ...' để client không bị đơ
        await sio.emit('action_update', {'action': "Lỗi Frame", 'confidence': 0.0})
        return

    action = "Đang chờ..."
    confidence = 0.0

    print("  -> Calling pose_extractor.extract()...") # <-- Print 2
    skeleton, _ = pose_extractor.extract(frame)
    print("  -> Finished pose_extractor.extract().") # <-- Print 3

    if skeleton is not None:
        print(f"  -> Detected skeleton! Adding to buffer (current size: {len(skeleton_buffer)})") # <-- Print 4
        skeleton_buffer.append(skeleton)
        print(f"  -> Buffer size after add: {len(skeleton_buffer)}") # <-- Print 5

        if len(skeleton_buffer) == 30:
            print("  --> Buffer full! Predicting...") # <-- Print 6
            action, confidence = recognizer.predict(list(skeleton_buffer))
            CONFIDENCE_THRESHOLD = 0.6
            if confidence < CONFIDENCE_THRESHOLD:
                action = "Unknown"
            print(f"  --> Prediction: {action} ({confidence:.2f})") # <-- Print 7
        else:
             print("  --> Buffer not full yet.") # <-- Print 8

    else:
        print("  -> No skeleton detected in this frame.") # <-- Print 9

    print(f"  -> Emitting action_update: {action}") # <-- Print 10
    await sio.emit('action_update', {
        'action': action,
        'confidence': round(confidence, 2)
    })
    
    # --- Chỉ xử lý AI, không vẽ ---
    skeleton, _ = pose_extractor.extract(frame) # Chỉ lấy skeleton, bỏ qua results

    if skeleton is not None:
        skeleton_buffer.append(skeleton)
        if len(skeleton_buffer) == 30:
            action, confidence = recognizer.predict(list(skeleton_buffer))
            CONFIDENCE_THRESHOLD = 0.6
            if confidence < CONFIDENCE_THRESHOLD:
                action = "Unknown"
    # --- Kết thúc xử lý AI ---

    # --- Chỉ gửi action và confidence ---
    await sio.emit('action_update', { # <-- Quay lại tên event cũ 'action_update'
        'action': action,
        'confidence': round(confidence, 2)
        # Không gửi 'image_data' nữa
    })
    # -----------------------------------

@sio.on('disconnect')
def disconnect(sid):
    print(f"Client disconnected: {sid}")

# --- KHỞI CHẠY SERVER ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)