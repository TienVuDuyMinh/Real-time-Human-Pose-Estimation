# File: src/pose_extractor.py

import cv2
import mediapipe as mp
import numpy as np
import sys

class MediaPipeExtractor:
    def __init__(self):
        """Khởi tạo MediaPipe Pose."""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            # Không cần mp_drawing nữa
        except Exception as e:
            print(f"[Extractor INIT] LỖI NGHIÊM TRỌNG khi khởi tạo MediaPipe: {e}")
            sys.exit(1)

    def extract(self, frame):
        """
        Trích xuất các khớp xương từ một khung hình.
        :param frame: Khung hình ảnh từ OpenCV (định dạng BGR).
        :return: Mảng numpy chứa tọa độ xương (shape: 25, 3) hoặc None nếu không phát hiện.
        """
        try:
            # Chuyển màu từ BGR sang RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Xử lý ảnh để phát hiện tư thế
            results = self.pose.process(image_rgb) # Chỉ cần chạy process

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Ánh xạ từ 33 khớp của MediaPipe sang 25 khớp của NTU
                coords = []
                # Joint 1: Spine base (ước lượng)
                coords.extend([(landmarks[24].x + landmarks[23].x) / 2, (landmarks[24].y + landmarks[23].y) / 2, (landmarks[24].z + landmarks[23].z) / 2])
                # Joint 2: Spine mid
                coords.extend([(landmarks[12].x + landmarks[11].x) / 2, (landmarks[12].y + landmarks[11].y) / 2, (landmarks[12].z + landmarks[11].z) / 2])
                # Joint 3: Neck
                coords.extend([(landmarks[10].x + landmarks[9].x) / 2, (landmarks[10].y + landmarks[9].y) / 2, (landmarks[10].z + landmarks[9].z) / 2])
                # Joint 4: Head
                coords.extend([landmarks[0].x, landmarks[0].y, landmarks[0].z])
                # Joint 5: Left shoulder
                coords.extend([landmarks[11].x, landmarks[11].y, landmarks[11].z])
                # Joint 6: Left elbow
                coords.extend([landmarks[13].x, landmarks[13].y, landmarks[13].z])
                # Joint 7: Left wrist
                coords.extend([landmarks[15].x, landmarks[15].y, landmarks[15].z])
                # Joint 8: Left hand
                coords.extend([landmarks[19].x, landmarks[19].y, landmarks[19].z])
                # Joint 9: Right shoulder
                coords.extend([landmarks[12].x, landmarks[12].y, landmarks[12].z])
                # Joint 10: Right elbow
                coords.extend([landmarks[14].x, landmarks[14].y, landmarks[14].z])
                # Joint 11: Right wrist
                coords.extend([landmarks[16].x, landmarks[16].y, landmarks[16].z])
                # Joint 12: Right hand
                coords.extend([landmarks[20].x, landmarks[20].y, landmarks[20].z])
                # Joint 13: Left hip
                coords.extend([landmarks[23].x, landmarks[23].y, landmarks[23].z])
                # Joint 14: Left knee
                coords.extend([landmarks[25].x, landmarks[25].y, landmarks[25].z])
                # Joint 15: Left ankle
                coords.extend([landmarks[27].x, landmarks[27].y, landmarks[27].z])
                # Joint 16: Left foot
                coords.extend([landmarks[31].x, landmarks[31].y, landmarks[31].z])
                # Joint 17: Right hip
                coords.extend([landmarks[24].x, landmarks[24].y, landmarks[24].z])
                # Joint 18: Right knee
                coords.extend([landmarks[26].x, landmarks[26].y, landmarks[26].z])
                # Joint 19: Right ankle
                coords.extend([landmarks[28].x, landmarks[28].y, landmarks[28].z])
                # Joint 20: Right foot
                coords.extend([landmarks[32].x, landmarks[32].y, landmarks[32].z])
                # Joint 21: Spine shoulder (giống joint 2)
                coords.extend([(landmarks[12].x + landmarks[11].x) / 2, (landmarks[12].y + landmarks[11].y) / 2, (landmarks[12].z + landmarks[11].z) / 2])
                # Joint 22: Left hand tip (dùng pinky)
                coords.extend([landmarks[17].x, landmarks[17].y, landmarks[17].z])
                # Joint 23: Left thumb
                coords.extend([landmarks[21].x, landmarks[21].y, landmarks[21].z])
                # Joint 24: Right hand tip (dùng pinky)
                coords.extend([landmarks[18].x, landmarks[18].y, landmarks[18].z])
                # Joint 25: Right thumb
                coords.extend([landmarks[22].x, landmarks[22].y, landmarks[22].z])

                # Chuyển đổi thành mảng numpy có shape (25, 3)
                skeleton = np.array(coords).reshape(25, 3)
                return skeleton # <-- Chỉ trả về skeleton
            
            return None # <-- Trả về None nếu không có xương
        except Exception as e:
            # print(f"    [Extractor] GẶP LỖI PYTHON: {e}") # Tắt log lỗi để đỡ rối
            return None # <-- Trả về None nếu có lỗi

    # Bỏ hàm draw_landmarks đi

    def close(self):
        """Giải phóng tài nguyên của MediaPipe Pose."""
        self.pose.close()