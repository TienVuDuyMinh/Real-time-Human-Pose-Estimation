# File: src/train.py (Phiên bản V3 - Lưu lịch sử Loss & Accuracy)
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json # Thêm thư viện json để lưu lịch sử

# --- Cấu hình Dự án ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

from src.models.stgcn import Model as STGCN

# --- CẤU HÌNH HUẤN LUYỆN ---
NTU_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'ntu_filtered_skeletons')
CUSTOM_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'custom_skeletons')
DATA_SOURCES = [NTU_DATA_DIR, CUSTOM_DATA_DIR]

PRETRAINED_WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', 'ntu60_stgcn.pt')
FINETUNED_WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', '20_last_epoch_model.pt')
BEST_WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', '20_best_finetuned_model.pt')
# --- THÊM: Đường dẫn lưu lịch sử ---
HISTORY_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', 'training_history.json')
# ---------------------------------

CLASS_MAPPING = {
    'walking': 0, 'running': 1, 'jumping': 2,
    'standing_up': 3, 'carrying': 4, 'lying_down': 5 
}
NUM_CLASSES = len(CLASS_MAPPING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_FRAMES = 300
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
# -----------------------------

# --- HÀM parse_skeleton_file (Giữ nguyên phiên bản V3 - Bật lại Log) ---
def parse_skeleton_file(filepath):
    # ... (Nội dung hàm parse_skeleton_file V3 giữ nguyên y hệt như cũ) ...
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        if not lines: tqdm.write(f"\n[Lỗi Parse] File {filepath} trống."); return None
        frame_count = int(lines[0].strip())
        frames_data = []; line_idx = 1; valid_frames_count = 0
        for i in range(frame_count):
            if line_idx + 1 >= len(lines): tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Thiếu dòng body count ở frame {i}. Dừng đọc file."); break
            try:
                body_count = int(lines[line_idx].strip()); line_idx += 1
                if body_count == 0:
                     temp_idx = line_idx; found_next_frame = False
                     while temp_idx < len(lines):
                          try: next_body_count = int(lines[temp_idx].strip()); line_idx = temp_idx; found_next_frame = True; break
                          except ValueError: temp_idx += 1
                     if not found_next_frame: line_idx = len(lines)
                     continue
                best_body_joints = None
                for j in range(body_count):
                    if line_idx + 1 >= len(lines): tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Thiếu dòng body info/joint count ở frame {i}, body {j}. Dừng đọc file."); line_idx = len(lines); break
                    try: line_idx += 1; joint_count = int(lines[line_idx].strip()); line_idx += 1
                    except (ValueError, IndexError): tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Lỗi đọc body info/joint count ở frame {i}, body {j}. Bỏ qua body."); line_idx += 25; continue
                    if line_idx + joint_count > len(lines): tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Thiếu dòng khớp ở frame {i}, body {j} (cần {joint_count}, còn {len(lines)-line_idx}). Dừng đọc file."); line_idx = len(lines); break
                    current_body_joints = []; joint_lines_read = 0
                    for k in range(joint_count):
                        try: joint_info = lines[line_idx].strip().split(); current_body_joints.append([float(coord) for coord in joint_info[:3]]); line_idx += 1; joint_lines_read += 1
                        except (ValueError, IndexError): tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Lỗi đọc khớp {k} ở frame {i}, body {j}. Bỏ qua body."); line_idx += (joint_count - joint_lines_read); current_body_joints = None; break
                    if j == 0 and current_body_joints is not None:
                        while len(current_body_joints) < 25: current_body_joints.append([0.0, 0.0, 0.0])
                        best_body_joints = np.array(current_body_joints[:25])
                if line_idx >= len(lines): break
                if best_body_joints is not None: frames_data.append(best_body_joints); valid_frames_count += 1
            except (ValueError, IndexError) as e_frame:
                 tqdm.write(f"\n[Cảnh báo Parse] File {filepath}: Lỗi đọc body count ở frame {i}: {e_frame}. Thử bỏ qua frame.")
                 temp_idx = line_idx + 1; found_next_frame = False
                 while temp_idx < len(lines):
                      try: next_body_count = int(lines[temp_idx].strip()); line_idx = temp_idx; found_next_frame = True; break
                      except ValueError: temp_idx += 1
                 if not found_next_frame: line_idx = len(lines)
                 if line_idx >= len(lines): break
                 continue
        if valid_frames_count > 0: return np.array(frames_data)
        else: tqdm.write(f"\n[Lỗi Dataset] File {filepath} không đọc được frame hợp lệ nào sau khi xử lý."); return None
    except Exception as e:
        tqdm.write(f"\n[Lỗi Dataset] Lỗi nghiêm trọng khi đọc file {filepath}: {e}")
        return None
# --- KẾT THÚC HÀM parse_skeleton_file ---

# --- CLASS SkeletonDataset ---
class SkeletonDataset(Dataset):
    def __init__(self, data_paths, class_map, max_frames):
        self.samples = []
        self.class_map = class_map
        self.max_frames = max_frames
        tqdm.write("Đang quét dữ liệu...")
        found_data = False
        for data_path in data_paths:
            if not os.path.exists(data_path):
                tqdm.write(f"[Cảnh báo Dataset] Không tìm thấy thư mục {data_path}. Bỏ qua.")
                continue
            found_in_source = False
            for class_name in os.listdir(data_path):
                class_dir = os.path.join(data_path, class_name)
                if class_name in self.class_map and os.path.isdir(class_dir):
                    class_label = self.class_map[class_name]
                    for file_name in os.listdir(class_dir):
                        if file_name.endswith('.skeleton'):
                            file_path = os.path.join(class_dir, file_name)
                            self.samples.append((file_path, class_label))
                            found_in_source = True; found_data = True
            if found_in_source: tqdm.write(f" -> Đã tìm thấy dữ liệu trong {os.path.basename(data_path)}")
        if found_data: tqdm.write(f"Tìm thấy tổng cộng {len(self.samples)} mẫu dữ liệu.")
        else: tqdm.write(f"[LỖI Dataset] Không tìm thấy bất kỳ file .skeleton nào.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]; data = parse_skeleton_file(file_path)
        if data is None or data.shape[0] == 0:
            tqdm.write(f"\n[Cảnh báo Dataset] Sử dụng dữ liệu zero cho file lỗi/rỗng: {file_path}")
            data = np.zeros((1, 25, 3))
        T, V, C = data.shape
        if V != 25 or C != 3:
             tqdm.write(f"\n[Cảnh báo Dataset] File {file_path} có shape sai ({data.shape}). Dùng zero.")
             data = np.zeros((1, 25, 3)); T, V, C = data.shape
        padded_data = np.zeros((self.max_frames, V, C), dtype=np.float32)
        if T >= self.max_frames: indices = np.linspace(0, T - 1, self.max_frames, dtype=int); padded_data = data[indices]
        else: padded_data[:T] = data
        data_final = np.transpose(padded_data, (2, 0, 1)); data_final = np.expand_dims(data_final, axis=-1)
        return torch.FloatTensor(data_final), label
# --- KẾT THÚC CLASS SkeletonDataset ---

def run_training():
    print(f"Bắt đầu quá trình fine-tuning trên thiết bị: {DEVICE}")
    print(f"Các lớp huấn luyện ({NUM_CLASSES}): {list(CLASS_MAPPING.keys())}")

    # 1. Tạo Dataset và Chia Train/Val 
    full_dataset = SkeletonDataset(DATA_SOURCES, CLASS_MAPPING, MAX_FRAMES)
    if len(full_dataset) == 0: print("LỖI: Không tìm thấy dữ liệu. Dừng lại."); return
    dataset_indices = list(range(len(full_dataset)))
    dataset_labels = [full_dataset.samples[i][1] for i in dataset_indices]

    train_indices, val_indices = train_test_split(
    dataset_indices,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    stratify=dataset_labels 
)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Đã chia dữ liệu: {len(train_dataset)} mẫu huấn luyện, {len(val_dataset)} mẫu kiểm định.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #train_indices, val_indices = train_test_split(dataset_indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=[full_dataset.samples[i][1] for i in dataset_indices])
    #print(f"Đã chia dữ liệu: {len(train_dataset)} train, {len(val_dataset)} val.")
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Tải kiến trúc Model 
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    model = STGCN(in_channels=3, num_class=60, graph_args=graph_args, edge_importance_weighting=True)

    # 3. Tải trọng số Pre-trained 
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"Đang tải trọng số pre-trained từ: {PRETRAINED_WEIGHTS_PATH}")
        try: model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, weights_only=True))
        except Exception as e: print(f"Lỗi tải pre-trained: {e}. Bắt đầu train từ đầu.")
    else: print(f"Không tìm thấy pre-trained tại {PRETRAINED_WEIGHTS_PATH}. Bắt đầu train từ đầu.")

    # 4. Thay thế lớp cuối cùng 
    try:
        in_features = model.fcn.in_channels
        model.fcn = nn.Conv2d(in_features, NUM_CLASSES, kernel_size=1)
        print(f"Đã thay thế lớp cuối cùng. Đầu ra mới: {NUM_CLASSES} lớp.")
    except AttributeError: print("LỖI: Không tìm thấy lớp 'fcn'."); return
    model.to(DEVICE)

    # 5. Thiết lập Optimizer và Loss 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- SỬA BƯỚC 6: THÊM TÍNH ACCURACY VÀ LƯU LỊCH SỬ ---
    print(f"\nBắt đầu huấn luyện {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    # --- THÊM: List để lưu lịch sử ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    # --------------------------------

    for epoch in range(NUM_EPOCHS):
        # --- Training phase ---
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        epoch_pbar = tqdm(total=len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)

        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            if output.dim() > 2 and output.shape[1] == 1: output = output.squeeze(1)
            if output.shape[0] != labels.shape[0]:
                 tqdm.write(f"\n[Lỗi Batch] Skip batch do shape mismatch."); epoch_pbar.update(1); continue

            try:
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # --- THÊM: Tính accuracy ---
                _, predicted = torch.max(output.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                # ------------------------
            except Exception as e: tqdm.write(f"\n[Lỗi Huấn luyện] {e}. Skip batch.")

            epoch_pbar.update(1)

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0

        # --- Validation phase ---
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                output = model(data)
                if output.dim() > 2 and output.shape[1] == 1: output = output.squeeze(1)
                if output.shape[0] != labels.shape[0]:
                     tqdm.write(f"\n[Lỗi Val] Skip batch do shape mismatch."); epoch_pbar.update(1); continue

                try:
                    loss = criterion(output, labels)
                    val_loss += loss.item()
                    # --- THÊM: Tính accuracy ---
                    _, predicted = torch.max(output.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    # ------------------------
                except Exception as e: tqdm.write(f"\n[Lỗi Kiểm định] {e}. Skip batch.")

                epoch_pbar.update(1)

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0

        epoch_pbar.close()
        # --- THÊM: In cả accuracy ---
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        # ----------------------------

        # --- THÊM: Lưu vào history ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        # ----------------------------

        # Lưu model tốt nhất (Giữ nguyên)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try: torch.save(model.state_dict(), BEST_WEIGHTS_PATH); print(f"   -> Val Loss cải thiện. Đã lưu best model: {BEST_WEIGHTS_PATH}")
            except Exception as e: print(f"   -> Lỗi lưu best model: {e}")
    # --- KẾT THÚC VÒNG LẶP EPOCH ---

    # 7. Lưu model cuối cùng (Giữ nguyên)
    print(f"\nHuấn luyện hoàn tất!")
    try: torch.save(model.state_dict(), FINETUNED_WEIGHTS_PATH); print(f"Model epoch cuối được lưu vào: {FINETUNED_WEIGHTS_PATH}")
    except Exception as e: print(f"Lỗi lưu model cuối: {e}")
    print(f"Model tốt nhất (Val Loss: {best_val_loss:.4f}) đã được lưu tại: {BEST_WEIGHTS_PATH}")

    # --- THÊM: Lưu history vào file JSON ---
    print(f"Đang lưu lịch sử huấn luyện vào: {HISTORY_PATH}")
    try:
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=4)
        print("Lưu lịch sử thành công.")
    except Exception as e:
        print(f"LỖI khi lưu lịch sử: {e}")
    # ---------------------------------------
    print("--- Kết thúc quá trình huấn luyện ---")

if __name__ == "__main__":
    run_training()