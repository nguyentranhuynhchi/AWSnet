#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường và chuẩn bị dữ liệu cho AWSnet trên Colab CLI
# Repository: https://github.com/nguyentranhuynhchi/AWSnet.git
# Protocol: myops380_train304_test76_last_v1 (No-Val, 300 epochs)
# ==============================================================================

set -e

echo "=== [1/5] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

# Tự động liên kết thư viện driver NVIDIA nếu có trên Colab (/usr/lib64-nvidia)
if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/5] Kiểm tra Mount Google Drive ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Drive chưa mount tự động (bỏ qua nếu tải dataset qua gdown):', e)
else:
    print('✅ Google Drive đã được mount sẵn tại /content/drive/MyDrive')
" || true

echo "=== [3/5] Cài đặt Dependencies cho AWSnet ==="
python3 -m pip install --upgrade pip -q
if [ -f "/content/AWSnet/requirements.txt" ]; then
    python3 -m pip install -r /content/AWSnet/requirements.txt -q
else
    python3 -m pip install torch torchvision numpy scipy h5py pandas nibabel tqdm gdown -q
fi

echo "=== [4/5] Tải Dataset MyoPS380 về SSD Cục bộ ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
    echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
    gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
        echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
    }
fi

for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
    echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
    command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
    unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
done

if [ -d "$DOWNLOAD_DIR/Processed_data" ]; then
    cp -r "$DOWNLOAD_DIR/Processed_data" "$TARGET_DATASET_DIR/"
fi

FOUND_TRAIN_NPZ=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "train_npz" 2>/dev/null | grep -i "bSSFP" | head -n 1 || true)
if [ -n "$FOUND_TRAIN_NPZ" ]; then
    LOCAL_DATASET=$(dirname "$(dirname "$FOUND_TRAIN_NPZ")")
    echo "✅ Đã xác định thư mục dataset chuẩn tại SSD: $LOCAL_DATASET"
else
    LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
    echo "⚠️ Chưa phát hiện thư mục bSSFP/train_npz, đặt mặc định: $LOCAL_DATASET"
fi

echo "=== [5/5] Kiểm tra PyTorch, tmux & Môi trường Hoàn tất ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "================================================================================"
echo "✅ THIẾT LẬP MÔI TRƯỜNG AWSNET HOÀN TẤT 100%!"
echo "Dataset chuẩn sẵn sàng tại: $LOCAL_DATASET"
echo "================================================================================"
