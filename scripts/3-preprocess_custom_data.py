# File: scripts/3-preprocess_custom_data.py
# (Phiên bản V3 - Có bộ đếm chẩn đoán)

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Thêm thư mục gốc vào Python Path để import các module từ src
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

try:
    from src.pose_extractor import MediaPipeExtractor
except ImportError:
    print("\nLỖI: Không thể import 'MediaPipeExtractor' từ 'src.pose_extractor'.")
    sys.exit(1)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'custom_videos')
OUTPUT_SKELETON_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'custom_skeletons')
# -------------------------

def format_skeleton_data(skeleton_frames):
    output_lines = []
    frame_count = len(skeleton_frames)
    output_lines.append(str(frame_count))
    for frame_data in skeleton_frames:
        output_lines.append("1")
        output_lines.append("0 0 0 0 0 0 0 0")
        output_lines.append(str(frame_data.shape[0]))
        for joint in frame_data:
            x, y, z = joint
            output_lines.append(f"{x} {y} {z} 0 0 0 0 0 0 0 0 2")
    return "\n".join(output_lines)

def process_videos():
    print(f"Bắt đầu xử lý video từ: {RAW_VIDEO_DIR}")
    print(f"Sẽ lưu file .skeleton vào: {OUTPUT_SKELETON_DIR}\n")
    
    try:
        extractor = MediaPipeExtractor()
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo MediaPipeExtractor. Lỗi: {e}")
        return

    try:
        class_folders = [f for f in os.listdir(RAW_VIDEO_DIR) if os.path.isdir(os.path.join(RAW_VIDEO_DIR, f))]
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy thư mục: {RAW_VIDEO_DIR}")
        return

    if not class_folders:
        print(f"LỖI: Không tìm thấy thư mục con (lớp hành động) nào trong {RAW_VIDEO_DIR}")
        return

    total_videos = 0
    total_files_generated = 0
    
    for class_name in class_folders:
        class_video_dir = os.path.join(RAW_VIDEO_DIR, class_name)
        class_skeleton_dir = os.path.join(OUTPUT_SKELETON_DIR, class_name)
        os.makedirs(class_skeleton_dir, exist_ok=True)
        
        video_files = [f for f in os.listdir(class_video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))]
        
        if not video_files:
            print(f"\n--- Lớp '{class_name}': Không tìm thấy file video nào.")
            continue
            
        print(f"\n--- Bắt đầu xử lý lớp: '{class_name}' ({len(video_files)} videos) ---")
        
        for video_name in tqdm(video_files, desc=f"Lớp {class_name}", unit="video"):
            total_videos += 1
            video_path = os.path.join(class_video_dir, video_name)
            skeleton_frames_for_video = []
            
            total_frames_in_video = 0
            skeletons_detected_in_video = 0
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    tqdm.write(f"\n[CẢNH BÁO] OpenCV không thể mở video: {video_name}. Bỏ qua.")
                    continue

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    total_frames_in_video += 1
                    skeleton = extractor.extract(frame)
                    
                    if skeleton is not None:
                        skeletons_detected_in_video += 1
                        skeleton_frames_for_video.append(skeleton)
                
                cap.release()

            except Exception as e:
                tqdm.write(f"\n[LỖI] Xử lý video {video_name} thất bại: {e}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                continue

            if total_frames_in_video == 0:
                tqdm.write(f"\n[CẢNH BÁO] Video {video_name} không có frame nào.")
            elif skeletons_detected_in_video == 0:
                tqdm.write(f"\n[CẢNH BÁO] Video {video_name}: Đã đọc {total_frames_in_video} frames, nhưng KHÔNG phát hiện được bộ xương nào.")
            else:
                tqdm.write(f" -> Video {video_name}: Đã đọc {total_frames_in_video} frames, phát hiện được {skeletons_detected_in_video} khung xương.")

            if skeleton_frames_for_video:
                try:
                    skeleton_file_content = format_skeleton_data(skeleton_frames_for_video)
                    skeleton_file_name = f"{os.path.splitext(video_name)[0]}_custom.skeleton"
                    output_path = os.path.join(class_skeleton_dir, skeleton_file_name)
                    
                    with open(output_path, 'w') as f:
                        f.write(skeleton_file_content)
                    total_files_generated += 1
                except Exception as e:
                    tqdm.write(f"\n[LỖI] Ghi file .skeleton cho {video_name} thất bại: {e}")

    print("\n\n--- HOÀN TẤT XỬ LÝ VIDEO ---")
    print(f"Đã duyệt qua tổng cộng {total_videos} video.")
    print(f"Đã tạo thành công {total_files_generated} file .skeleton.")

if __name__ == "__main__":
    process_videos()