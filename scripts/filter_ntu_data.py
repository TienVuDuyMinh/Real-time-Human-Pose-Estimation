import os
import shutil
from tqdm import tqdm

# --- !!! THAY ĐỔI 2 DÒNG NÀY !!! ---
# 1. Đường dẫn đến thư mục chứa toàn bộ file .skeleton của NTU
SOURCE_DIR = r"D:\code_etc\Python\_File_code\Pose_estimation_Final\data\raw\ntu_rgbd_skeletons\nturgb+d_skeletons" 

# 2. Đường dẫn đến thư mục bạn muốn lưu dữ liệu đã lọc
DEST_DIR = r"D:\code_etc\Python\_File_code\Pose_estimation_Final\data\processed\ntu_filtered_skeletons"
# ------------------------------------

# Ánh xạ từ tên lớp tùy chỉnh sang danh sách các Action ID của NTU
ACTION_MAPPING = {
    'walking': ['A006'],
    'running': ['A007'],
    'jumping': ['A009'],
    'standing_up': ['A023'], # Dùng hành động "đứng dậy" để đại diện cho "đứng thẳng"
    'carrying': ['A058'],
    'lying_down': ['A043']
}

def filter_skeleton_files():
    """
    Lọc và sao chép các file skeleton từ thư mục nguồn sang thư mục đích,
    sắp xếp chúng vào các thư mục con theo từng lớp hành động.
    """
    print(f"Bắt đầu quá trình lọc dữ liệu từ: {SOURCE_DIR}")
    
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Đã tạo thư mục đích: {DEST_DIR}")

    # Tạo các thư mục con cho từng lớp hành động
    for class_name in ACTION_MAPPING.keys():
        class_path = os.path.join(DEST_DIR, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    # Tạo một mapping ngược để tra cứu tên lớp từ Action ID cho hiệu quả
    id_to_class_map = {}
    for class_name, action_ids in ACTION_MAPPING.items():
        for action_id in action_ids:
            id_to_class_map[action_id] = class_name
            
    # Lấy danh sách tất cả các file skeleton trong thư mục nguồn
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.skeleton')]
    
    if not all_files:
        print(f"LỖI: Không tìm thấy file .skeleton nào trong '{SOURCE_DIR}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    print(f"Tìm thấy {len(all_files)} file. Bắt đầu lọc...")

    # Sử dụng tqdm để hiện thanh tiến trình
    copied_count = 0
    for filename in tqdm(all_files, desc="Đang xử lý các file"):
        try:
            # Tên file NTU có định dạng: SsssCcccPpppRrrrAaaa
            # Action ID nằm ở vị trí Aaaa, ví dụ A006
            action_id = filename[16:20] # Lấy chuỗi từ ký tự 17 đến 20

            if action_id in id_to_class_map:
                # Lấy tên lớp tương ứng
                class_name = id_to_class_map[action_id]
                
                # Tạo đường dẫn nguồn và đích
                source_path = os.path.join(SOURCE_DIR, filename)
                dest_path = os.path.join(DEST_DIR, class_name, filename)
                
                # Sao chép file
                shutil.copy2(source_path, dest_path)
                copied_count += 1
        except Exception as e:
            print(f"\nLỗi khi xử lý file {filename}: {e}")
            
    print("\n--- HOÀN TẤT ---")
    print(f"Tổng cộng đã sao chép {copied_count} file vào thư mục '{DEST_DIR}'.")
    
    # In ra thống kê số file mỗi lớp
    print("\nThống kê số lượng file mỗi lớp:")
    for class_name in ACTION_MAPPING.keys():
        class_path = os.path.join(DEST_DIR, class_name)
        num_files = len(os.listdir(class_path))
        print(f"- {class_name}: {num_files} file")


if __name__ == "__main__":
    filter_skeleton_files()