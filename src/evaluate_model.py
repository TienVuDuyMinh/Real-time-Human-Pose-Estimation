# File: src/evaluate_model.py 
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
from tqdm import tqdm 

# --- Cấu hình Dự án ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR)

# Import Dataset và Model từ train.py và models/
try:
    from src.train import SkeletonDataset, CLASS_MAPPING, MAX_FRAMES, RANDOM_SEED, VALIDATION_SPLIT, DATA_SOURCES # Lấy các cấu hình từ train.py
    from src.models.stgcn import Model as STGCN
except ImportError as e:
    print(f"LỖI: Không thể import từ src/train.py hoặc src/models/. Lỗi: {e}")
    sys.exit(1)

# --- CẤU HÌNH ĐÁNH GIÁ ---
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', 'best_finetuned_model.pt')
HISTORY_PATH = os.path.join(PROJECT_ROOT_DIR, 'weights', 'training_history.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'evaluation_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_CLASSES = len(CLASS_MAPPING)
CLASS_NAMES = list(CLASS_MAPPING.keys())
# -----------------------------

def plot_training_history(history, save_path):
    """Vẽ biểu đồ loss và accuracy từ history."""
    if not history: return # Thoát nếu không có history
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if not epochs: return # Thoát nếu history rỗng

    plt.figure(figsize=(12, 5))
    try:
        # Vẽ Loss
        plt.subplot(1, 2, 1)
        if 'train_loss' in history and 'val_loss' in history:
            plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
            plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        else: plt.text(0.5, 0.5, 'Loss data missing', ha='center', va='center')

        # Vẽ Accuracy
        plt.subplot(1, 2, 2)
        if 'train_acc' in history and 'val_acc' in history:
            plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
            plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
        else: plt.text(0.5, 0.5, 'Accuracy data missing', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ lịch sử huấn luyện vào: {save_path}")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ lịch sử: {e}")
    # plt.show()

def plot_confusion_matrix(cm, class_names, save_path):
    """Vẽ confusion matrix dưới dạng heatmap."""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Đã lưu Confusion Matrix vào: {save_path}")
    except Exception as e:
        print(f"Lỗi khi vẽ Confusion Matrix: {e}")
    # plt.show()


def evaluate_model():
    print(f"Bắt đầu đánh giá model trên thiết bị: {DEVICE}")
    print(f"Sử dụng model: {MODEL_WEIGHTS_PATH}")

    # 1. Tải Lịch sử Huấn luyện
    history = None
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f: history = json.load(f)
            print("Tải lịch sử huấn luyện thành công.")
            plot_training_history(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))
        except Exception as e: print(f"Lỗi tải/vẽ lịch sử: {e}")
    else: print(f"Không tìm thấy file lịch sử: {HISTORY_PATH}")

    # 2. Tạo Dataset và DataLoader cho tập Validation
    print("Đang tải dữ liệu kiểm định...")
    try:
        full_dataset = SkeletonDataset(DATA_SOURCES, CLASS_MAPPING, MAX_FRAMES)
        if len(full_dataset) == 0: print("LỖI: Không tìm thấy dữ liệu. Dừng lại."); return
        dataset_indices = list(range(len(full_dataset)))
        _, val_indices = train_test_split(dataset_indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=[full_dataset.samples[i][1] for i in dataset_indices])
        val_dataset = Subset(full_dataset, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"Đã tải {len(val_dataset)} mẫu kiểm định.")
    except Exception as e:
        print(f"LỖI nghiêm trọng khi tải dữ liệu: {e}")
        return

    # 3. Tải Model đã Fine-tune
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    model = STGCN(in_channels=3, num_class=NUM_CLASSES, graph_args=graph_args, edge_importance_weighting=True)
    try:
        # Thêm weights_only=True để tắt cảnh báo
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device(DEVICE), weights_only=True))
        model.to(DEVICE)
        model.eval()
        print("Tải model fine-tuned thành công.")
    except FileNotFoundError: print(f"LỖI: Không tìm thấy file model tại {MODEL_WEIGHTS_PATH}."); return
    except Exception as e: print(f"LỖI khi tải model: {e}."); return

    # 4. Chạy dự đoán trên tập Validation
    print("Đang chạy dự đoán trên tập kiểm định...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Evaluating"):
            try:
                data = data.to(DEVICE)
                output = model(data)
                if output.dim() > 2 and output.shape[1] == 1: output = output.squeeze(1)

                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                 tqdm.write(f"\n[Lỗi Inference] Gặp lỗi trong batch: {e}. Bỏ qua batch.")

    if not all_labels or not all_preds:
        print("LỖI: Không có kết quả dự đoán nào được tạo ra. Dừng đánh giá.")
        return

    # 5. Tính toán và Hiển thị Kết quả
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")

    # Classification Report (Đã sửa lỗi ValueError)
    print("\nClassification Report:")
    try:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            labels=range(NUM_CLASSES), # <-- Thêm labels
            zero_division=0           # <-- Thêm zero_division
        )
        print(report)
        # Lưu report vào file
        report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f: f.write(report)
        print(f"Đã lưu Classification Report vào: {report_path}")
    except Exception as e:
        print(f"Lỗi khi tạo Classification Report: {e}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    try:
        cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES)) # Thêm labels ở đây nữa
        print(cm)
        # Vẽ và lưu confusion matrix
        plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    except Exception as e:
        print(f"Lỗi khi tạo/vẽ Confusion Matrix: {e}")

    print("\n--- Kết thúc đánh giá ---")

if __name__ == "__main__":
    # Tự động cài thư viện nếu thiếu
    try: import seaborn
    except ImportError:
        print("\nĐang cài đặt seaborn..."); os.system(f"{sys.executable} -m pip install seaborn matplotlib")
        import seaborn as sns; import matplotlib.pyplot as plt

    evaluate_model()