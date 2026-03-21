import json
import subprocess
from pathlib import Path
from ultralytics import YOLO
import sys

def sweep():
    base_dir = Path("/home/me/ht/nmiai/tasks/object-detection")
    model_path = base_dir / "yolo-approach/best.pt"
    images_dir = base_dir / "data-creation/data/clean_split/val/images"
    labels_dir = base_dir / "data-creation/data/clean_split/val/labels"
    dataset_yaml = base_dir / "data-creation/data/clean_split/dataset.yaml"
    eval_script = base_dir / "vlm-approach/eval_stratified_map.py"
    output_json = Path("/tmp/sweep_predictions.json")
    
    print(f"Loading model {model_path}...")
    model = YOLO(str(model_path), task="detect")
    
    # We will do Confidence threshold sweep
    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    best_conf = 0
    best_map = 0
    
    for conf_thresh in thresholds:
        print(f"\n--- Running inference with Confidence = {conf_thresh:.4f} ---")
        results = model.predict(
            source=str(images_dir),
            iou=0.35,        # using 0.35 based on earlier sweep
            conf=conf_thresh,
            max_det=300,     # max det default
            verbose=False,
            device="cuda:0"
        )
        
        predictions = []
        for r in results:
            image_id = Path(r.path).name
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Extract boxes in xyxy format to convert to xywh (x_min, y_min, width, height)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1
                predictions.append({
                    "image_id": image_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "category_id": int(cls[i]),
                    "score": float(confs[i])
                })
        
        output_json.write_text(json.dumps(predictions))
        
        # Create a dummy submission directory that just copies the predictions
        dummy_dir = Path("/tmp/dummy_submission")
        dummy_dir.mkdir(parents=True, exist_ok=True)
        (dummy_dir / "run.py").write_text("""import argparse
import shutil
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()
shutil.copy('/tmp/sweep_predictions.json', args.output)
""")
        
        # Run eval_stratified_map.py
        cmd = [
            sys.executable,
            str(eval_script),
            "--submission-dir", str(dummy_dir),
            "--images", str(images_dir),
            "--labels", str(labels_dir),
            "--dataset-yaml", str(dataset_yaml),
            "--output-json", "/tmp/eval_output.json"
        ]
        
        result_proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            # The script prints the result as JSON to stdout
            eval_result = json.loads(result_proc.stdout)
            det_map = eval_result["detection_map50"]
            print(f"Result for Conf={conf_thresh:.4f}: Detection mAP@0.5 = {det_map:.4f}")
            if det_map > best_map:
                best_map = det_map
                best_conf = conf_thresh
        except Exception as e:
            print("Failed to parse evaluation output:", e)
            print("Stdout:", result_proc.stdout)
            print("Stderr:", result_proc.stderr)
            
    print(f"\nBest Confidence threshold: {best_conf:.4f} with Detection mAP@0.5 = {best_map:.4f}")

if __name__ == "__main__":
    sweep()
