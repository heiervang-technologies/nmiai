FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.10 (Ubuntu 22.04 default) + essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    unzip file \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Match sandbox package versions EXACTLY
# Step 1: PyTorch + torchvision with CUDA 12.4
RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: All other packages pinned to sandbox versions
RUN pip3 install --no-cache-dir \
    ultralytics==8.1.0 \
    onnxruntime-gpu==1.20.0 \
    "opencv-python-headless==4.9.0.80" \
    albumentations==1.3.1 \
    Pillow==10.2.0 \
    "numpy==1.26.4" \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    pycocotools==2.0.7 \
    ensemble-boxes==1.0.9 \
    timm==0.9.12 \
    supervision==0.18.0 \
    safetensors==0.4.2

# Copy the sandbox runner + validator
COPY sandbox_runner.py /sandbox_runner.py
COPY validate_output.py /validate_output.py

# Working directory for submission extraction
RUN mkdir -p /submission /data/images

ENTRYPOINT ["python3", "/sandbox_runner.py"]
