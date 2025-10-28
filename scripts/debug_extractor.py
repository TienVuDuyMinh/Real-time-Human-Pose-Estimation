# File: scripts/debug_extractor.py
import sys
import os

# Thêm thư mục gốc vào Python Path
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

print("--- Đang thử import 'MediaPipeExtractor'...")
try:
    from src.pose_extractor import MediaPipeExtractor
    print("--- Import 'MediaPipeExtractor' THÀNH CÔNG.")
except Exception as e:
    print(f"--- LỖI khi import: {e}")
    sys.exit()

print("\n--- Đang thử KHỞI TẠO 'MediaPipeExtractor'...")
try:
    extractor = MediaPipeExtractor()
    print("--- KHỞI TẠO extractor THÀNH CÔNG.")
except Exception as e:
    print(f"--- LỖI khi khởi tạo: {e}")

print("\n--- KẾT THÚC CHẨN ĐOÁN ---")