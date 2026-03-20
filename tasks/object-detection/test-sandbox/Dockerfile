FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip unzip && rm -rf /var/lib/apt/lists/*

# Match sandbox package versions exactly
RUN pip3 install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    ultralytics==8.1.0 \
    onnxruntime-gpu==1.20.0 \
    opencv-python-headless==4.9.0.80 \
    albumentations==1.3.1 \
    Pillow==10.2.0 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    pycocotools==2.0.7 \
    ensemble-boxes==1.0.9 \
    timm==0.9.12 \
    supervision==0.18.0 \
    safetensors==0.4.2

WORKDIR /submission
