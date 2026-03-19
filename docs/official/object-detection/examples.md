# Object Detection - Examples

## Random Baseline

A simple random prediction baseline to verify submission format:

```python
import json
import pathlib
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))

    predictions = []
    for img_path in image_files:
        image_id = int(img_path.stem)
        num_detections = random.randint(1, 10)
        for _ in range(num_detections):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 355),
                "bbox": [
                    random.uniform(0, 500),
                    random.uniform(0, 500),
                    random.uniform(10, 200),
                    random.uniform(10, 200),
                ],
                "score": random.uniform(0.0, 1.0),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Generated {len(predictions)} predictions for {len(image_files)} images.")


if __name__ == "__main__":
    main()
```

## YOLOv8 Approach

### Pretrained COCO Model

A pretrained YOLOv8 model trained on COCO will output **wrong category IDs** since COCO has 80 classes, not 356. You must fine-tune the model on the competition dataset.

### Fine-Tuning

When fine-tuning YOLOv8 on the competition dataset, set:

```python
# Set number of classes to 357 (0-356 inclusive, but nc = count)
# Adjust based on whether categories are 0-indexed or 1-indexed
model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50, nc=357)
```

Ensure your `dataset.yaml` maps category IDs correctly to the competition's 356 categories.

## ONNX Export and Inference

### Export

When exporting to ONNX, use **opset version <= 20**:

```python
model.export(format="onnx", opset=20)
```

### ONNX Inference

Use `CUDAExecutionProvider` for GPU acceleration:

```python
import onnxruntime as ort
import numpy as np
import json
import pathlib
import argparse


def preprocess(image_path, input_size=(640, 640)):
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(input_size)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def postprocess(outputs, image_id, score_threshold=0.25):
    predictions = []
    # Adjust based on your model's output format
    detections = outputs[0]
    for det in detections:
        score = float(det[4])
        if score < score_threshold:
            continue
        x, y, w, h = det[0], det[1], det[2], det[3]
        category_id = int(det[5])
        predictions.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": score,
        })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    session = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))

    all_predictions = []
    for img_path in image_files:
        image_id = int(img_path.stem)
        img_tensor = preprocess(img_path)
        outputs = session.run(None, {input_name: img_tensor})
        preds = postprocess(outputs, image_id)
        all_predictions.extend(preds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)

    print(f"Generated {len(all_predictions)} predictions for {len(image_files)} images.")


if __name__ == "__main__":
    main()
```
