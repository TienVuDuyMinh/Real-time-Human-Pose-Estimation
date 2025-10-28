# File: src/inference.py

import torch
import numpy as np
import os
import sys

# Thêm thư mục gốc của dự án vào Python Path để có thể import src
# Điều này giúp bạn chạy file này độc lập để kiểm tra
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

# Import kiến trúc model từ file bạn đã tạo
from src.models.stgcn import Model as STGCN # Chú ý: Tên lớp có thể là 'Model' hoặc 'STGCN' tùy vào file bạn tải

class ActionRecognizer:
    def __init__(self, model_weights_path):
        """
        Khởi tạo bộ nhận dạng hành động.
        :param model_weights_path: Đường dẫn đến file trọng số .pt của model.
        """
        # --- CẤU HÌNH ---
        # Danh sách các lớp phải khớp với thứ tự trong dữ liệu của bạn và model
        # Đây là 60 lớp của NTU-60. Chúng ta sẽ lọc ra sau.
        # Code mới (6 lớp - kiểm tra lại thứ tự nếu cần)
        self.class_names = [
            'walking', 
            'running', 
            'jumping', 
            'standing_up', 
            'carrying', 
            'lying_down' 
        ]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")

        # --- TẢI MODEL ---
        # Các tham số này phải khớp với model bạn tải về
        graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        
        # Tải kiến trúc model
        self.model = STGCN(in_channels=3, num_class=6, graph_args=graph_args,
                           edge_importance_weighting=True).to(self.device)
        
        # Tải trọng số đã huấn luyện
        try:
            self.model.load_state_dict(torch.load(model_weights_path))
            print("Tải trọng số model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải trọng số: {e}")
            print("Vui lòng kiểm tra lại đường dẫn và kiến trúc model.")

        # Chuyển model sang chế độ đánh giá
        self.model.eval()

    def preprocess_data(self, skeleton_data):
        """
        Tiền xử lý dữ liệu xương để phù hợp với đầu vào của model.
        :param skeleton_data: Mảng numpy chứa dữ liệu xương của một chuỗi hành động.
                              Shape: (số_frame, số_khớp, số_tọa_độ)
        :return: Tensor sẵn sàng để đưa vào model.
        """
        # Model ST-GCN yêu cầu đầu vào có shape: (N, C, T, V, M)
        # N: batch size (1), C: kênh (3), T: frames, V: khớp (25), M: người (1)
        
        data = skeleton_data
        T, V, C = data.shape
        
        # Chuyển đổi shape: (T, V, C) -> (C, T, V)
        data = np.transpose(data, (2, 0, 1))
        
        # Thêm các chiều N và M: (C, T, V) -> (1, C, T, V, 1)
        data = data[np.newaxis, :, :, :, np.newaxis]
        
        # Chuyển thành tensor
        data_tensor = torch.from_numpy(data).float().to(self.device)
        
        return data_tensor

    def predict(self, skeleton_sequence):
        """
        Dự đoán hành động từ một chuỗi dữ liệu xương.
        :param skeleton_sequence: Dữ liệu xương đã được parse.
        :return: Tên hành động dự đoán và độ tin cậy.
        """
        if not skeleton_sequence:
            return "Không có dữ liệu", 0.0

        body_data = np.array([frame[0] for frame in skeleton_sequence])
        body_data_3d = body_data[:, :, :3]

        input_tensor = self.preprocess_data(body_data_3d)

        with torch.no_grad():
            output = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0, 0], dim=0)
        confidence, predicted_index = torch.max(probabilities, 0)
        
        predicted_class_name = self.class_names[predicted_index.item()]
        
        return predicted_class_name, confidence.item()

# --- CODE KIỂM TRA NHANH ---
# Bạn có thể chạy file này trực tiếp bằng lệnh: python src/inference.py
# để kiểm tra xem mọi thứ có hoạt động không.
def parse_skeleton_file(filepath):
    """Hàm parse file .skeleton (sao chép từ notebook để kiểm tra)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        frame_count = int(lines[0])
        frames_data = []
        line_idx = 1
        for i in range(frame_count):
            body_count = int(lines[line_idx])
            bodies_data = []
            line_idx += 1
            for j in range(body_count):
                line_idx += 1
                joint_count = int(lines[line_idx])
                joints_data = []
                line_idx += 1
                for k in range(joint_count):
                    joint_info = lines[line_idx].split()
                    joint_data = [float(coord) for coord in joint_info]
                    joints_data.append(joint_data)
                    line_idx += 1
                bodies_data.append(np.array(joints_data))
            frames_data.append(bodies_data)
    return frames_data

if __name__ == '__main__':
    # Đường dẫn đến file trọng số bạn đã tải
    # !!! THAY ĐỔI TÊN FILE NÀY cho đúng với file của bạn !!!
    weights_filename = 'ntu60_stgcn.pt'
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', weights_filename)
    
    # 1. Khởi tạo bộ nhận dạng
    recognizer = ActionRecognizer(WEIGHTS_PATH)
    
    # 2. Tải và parse một file skeleton mẫu
    SAMPLE_FILE = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'ntu_filtered_skeletons', 
                               'running', 'S001C001P002R002A007.skeleton')

    skeleton_data = parse_skeleton_file(SAMPLE_FILE)
    
    # 3. Dự đoán
    if skeleton_data:
        action, confidence = recognizer.predict(skeleton_data)
        print(f"\n---> Kết quả dự đoán: '{action}' (Độ tin cậy: {confidence:.2f})")