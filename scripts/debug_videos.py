# File: scripts/debug_videos.py
import os
import cv2

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'custom_videos')

print("--- BẮT ĐẦU CHẨN ĐOÁN VIDEO ---")
print(f"Sẽ tìm video trong: {RAW_VIDEO_DIR}\n")

found_video = False
first_video_path = ""
first_video_name = ""

# Tìm video ĐẦU TIÊN để kiểm tra
for class_folder in os.listdir(RAW_VIDEO_DIR):
    class_folder_path = os.path.join(RAW_VIDEO_DIR, class_folder)
    if os.path.isdir(class_folder_path):
        for video_name in os.listdir(class_folder_path):
            if video_name.endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                first_video_path = os.path.join(class_folder_path, video_name)
                first_video_name = video_name
                found_video = True
                break
    if found_video:
        break

if not found_video:
    print("!!! LỖI: Không tìm thấy bất kỳ file video nào trong các thư mục con.")
    print("Vui lòng kiểm tra lại cấu trúc thư mục data/raw/custom_videos/")
    sys.exit()

print(f"Đang thử mở video: {first_video_name} tại {first_video_path}")

try:
    cap = cv2.VideoCapture(first_video_path)
    
    if not cap.isOpened():
        print("\n--- KẾT QUẢ: THẤT BẠI ---")
        print(f"LỖI: OpenCV (cv2) không thể mở file video này.")
        print("Đây là vấn đề về codec video.")
    else:
        print("OpenCV đã mở video thành công.")
        ret, frame = cap.read()
        
        if ret:
            print(f"Đã đọc thành công 1 khung hình (kích thước: {frame.shape})")
            print("\n--- KẾT QUẢ: THÀNH CÔNG ---")
            print("Vấn đề không nằm ở việc đọc video.")
        else:
            print("\n--- KẾT QUẢ: THẤT BẠI ---")
            print(f"LỖI: Có thể mở video, nhưng không thể đọc khung hình (ret=False).")
            print("File video có thể bị hỏng hoặc có định dạng không được hỗ trợ.")
    
    cap.release()

except Exception as e:
    print(f"\n--- KẾT QUẢ: THẤT BẠI (Lỗi nghiêm trọng) ---")
    print(f"Đã xảy ra lỗi bất ngờ: {e}")

print("\n--- KẾT THÚC CHẨN ĐOÁN ---")