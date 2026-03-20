"""
Automated overnight checkpoint evaluation pipeline.

Watches for new checkpoints from the trainer, then:
1. NF4 quantize
2. Test with run_fast.py on val set
3. Compute mAP@0.5 (detection + classification)
4. Compare to DINOv2 baseline (92% cls mAP)
5. Package best submission ZIP
6. Report results

Usage: CUDA_VISIBLE_DEVICES=0 uv run python overnight_pipeline.py

Can also be run on a specific checkpoint:
  uv run python overnight_pipeline.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import json
import functools
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

print = functools.partial(print, flush=True)

# === PATHS ===
WATCH_DIRS = [
    Path(__file__).parent / "training_output_multitask",
    Path(__file__).parent / "training_output",
]
PRUNED_DIR = Path(__file__).parent / "pruned"
EXPORT_DIR = Path(__file__).parent / "exported"
SUBMISSION_DIR = Path(__file__).parent.parent / "submission-markusnet"
VAL_IMGS = Path(__file__).parent.parent / "data-creation" / "data" / "stratified_split" / "val" / "images"
VAL_LABS = Path(__file__).parent.parent / "data-creation" / "data" / "stratified_split" / "val" / "labels"
YOLO_ONNX = Path(__file__).parent.parent / "yolo-approach" / "best.onnx"
RESULTS_LOG = Path(__file__).parent / "overnight_results.json"
BASELINE_CLS_MAP = 0.9199  # DINOv2 baseline
BASELINE_COMBINED = 0.8000  # DINOv2 combined

# NF4 table
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float32)
GROUP_SIZE = 64


def nf4_quantize_checkpoint(ckpt_path, output_path):
    """NF4 quantize a checkpoint, stripping embed_tokens + lm_head."""
    print(f"  NF4 quantizing {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    nf4_state = {}
    fp16_state = {}

    for k, v in ckpt["model_state"].items():
        if "embed_tokens" in k or "lm_head" in k:
            continue
        if v.dim() >= 2 and v.numel() >= GROUP_SIZE:
            flat = v.float().reshape(-1)
            n = flat.numel()
            pad_n = (GROUP_SIZE - n % GROUP_SIZE) % GROUP_SIZE
            if pad_n > 0:
                flat = torch.cat([flat, torch.zeros(pad_n)])
            groups = flat.reshape(-1, GROUP_SIZE)
            absmax = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = groups / absmax
            distances = (normalized.unsqueeze(-1) - NF4_TABLE.unsqueeze(0).unsqueeze(0)).abs()
            indices = distances.argmin(dim=-1).reshape(-1)
            packed = (indices[0::2].to(torch.uint8) << 4) | indices[1::2].to(torch.uint8)
            nf4_state[k] = {
                "packed": packed, "scales": absmax.squeeze(1).half(),
                "shape": v.shape, "numel": n,
            }
        else:
            fp16_state[k] = v.half() if v.is_floating_point() else v

    cls_state = {k: v.half() for k, v in ckpt["cls_head_state"].items()}

    torch.save({
        "nf4_state": nf4_state, "fp16_state": fp16_state,
        "cls_head_state": cls_state,
        "accuracy": ckpt.get("accuracy", 0),
        "global_step": ckpt.get("global_step", 0),
        "quantization": "nf4", "group_size": GROUP_SIZE,
    }, output_path)

    size = output_path.stat().st_size / 1024**2
    print(f"  NF4 saved: {size:.0f} MB (acc={ckpt.get('accuracy', 0):.4f})")
    return size


def run_inference(nf4_path, val_imgs, output_json):
    """Run MarkusNet inference on val images using run_fast.py."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    # Import run_fast from submission dir
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_fast", str(SUBMISSION_DIR / "run_fast.py"))
    run_fast = importlib.util.module_from_spec(spec)

    # We can't directly import run_fast due to __name__ guard
    # Instead, use the YOLO detection + our own MarkusNet classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(YOLO_ONNX), providers=providers)
    inp_info = sess.get_inputs()[0]
    inp_name = inp_info.name
    H, W = inp_info.shape[2], inp_info.shape[3]

    # Load NF4 MarkusNet
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR), dtype=torch.bfloat16,
        ignore_mismatched_sizes=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # Dequantize NF4
    ckpt = torch.load(nf4_path, map_location=device, weights_only=False)
    state = {}
    for k, q in ckpt["nf4_state"].items():
        packed = q["packed"].to(device)
        scales = q["scales"].to(device)
        high = (packed >> 4).long()
        low = (packed & 0x0F).long()
        indices = torch.stack([high, low], dim=1).reshape(-1)
        nf4 = NF4_TABLE.to(device)
        values = nf4[indices].reshape(-1, GROUP_SIZE) * scales.float().unsqueeze(1)
        state[k] = values.reshape(-1)[:q["numel"]].reshape(q["shape"]).half()
    for k, v in ckpt["fp16_state"].items():
        state[k] = v.to(device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    class ClsHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(1024,1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024,356))
        def forward(self, x): return self.head(x.mean(dim=1))

    cls_head = ClsHead()
    cls_head.load_state_dict(ckpt["cls_head_state"])
    cls_head.to(device).eval()

    # YOLO detection helpers
    def letterbox(img, new_shape):
        h, w = img.shape[:2]
        r = min(new_shape[0]/h, new_shape[1]/w)
        nw, nh = int(round(w*r)), int(round(h*r))
        dw, dh = (new_shape[1]-nw)/2, (new_shape[0]-nh)/2
        resized = cv2.resize(img, (nw, nh))
        lb = cv2.copyMakeBorder(resized, int(round(dh-0.1)), int(round(dh+0.1)),
                                int(round(dw-0.1)), int(round(dw+0.1)),
                                cv2.BORDER_CONSTANT, value=(114,114,114))
        return lb, r, (dw, dh)

    def nms(boxes, scores, thresh):
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order)>0:
            i = order[0]; keep.append(i)
            if len(order)==1: break
            xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
            iou=inter/(areas[i]+areas[order[1:]]-inter+1e-8)
            order=order[np.where(iou<=thresh)[0]+1]
        return keep

    results = []
    img_paths = sorted(val_imgs.glob("*.jpg"))

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        oh, ow = img_bgr.shape[:2]

        # YOLO detect
        lb, ratio, (dw, dh) = letterbox(img_bgr, (H, W))
        tensor = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        tensor = np.transpose(tensor, (2,0,1))[None]
        out = sess.run(None, {inp_name: tensor})[0][0].T
        boxes_cxcywh, scores_all = out[:,:4], out[:,4:]
        cls_ids = np.argmax(scores_all, 1)
        confs = np.array([scores_all[j, cls_ids[j]] for j in range(len(cls_ids))])
        mask = confs > 0.001
        boxes_cxcywh, cls_ids, confs = boxes_cxcywh[mask], cls_ids[mask], confs[mask]
        if len(boxes_cxcywh) == 0: continue

        xyxy = np.zeros_like(boxes_cxcywh)
        xyxy[:,0]=(boxes_cxcywh[:,0]-boxes_cxcywh[:,2]/2-dw)/ratio
        xyxy[:,1]=(boxes_cxcywh[:,1]-boxes_cxcywh[:,3]/2-dh)/ratio
        xyxy[:,2]=(boxes_cxcywh[:,0]+boxes_cxcywh[:,2]/2-dw)/ratio
        xyxy[:,3]=(boxes_cxcywh[:,1]+boxes_cxcywh[:,3]/2-dh)/ratio
        xyxy[:,[0,2]]=np.clip(xyxy[:,[0,2]],0,ow)
        xyxy[:,[1,3]]=np.clip(xyxy[:,[1,3]],0,oh)

        # Per-class NMS
        fb, fs, fc = [], [], []
        for c in range(356):
            m = cls_ids==c
            if not np.any(m): continue
            cb, cs = xyxy[m], confs[m]
            keep = nms(cb, cs, 0.45)
            fb.append(cb[keep]); fs.append(cs[keep]); fc.extend([c]*len(keep))
        if not fb: continue
        fb, fs = np.concatenate(fb), np.concatenate(fs)
        if len(fs)>300:
            top=np.argsort(fs)[::-1][:300]
            fb,fs,fc=fb[top],fs[top],[fc[j] for j in top]

        # MarkusNet classify crops
        img_pil = Image.open(img_path).convert("RGB")
        crops = []
        for x1,y1,x2,y2 in fb:
            x1i,y1i,x2i,y2i=max(0,int(x1)),max(0,int(y1)),min(ow,int(x2)),min(oh,int(y2))
            if x2i>x1i and y2i>y1i:
                crops.append(img_pil.crop((x1i,y1i,x2i,y2i)))
            else:
                crops.append(Image.new("RGB",(32,32),(128,128,128)))

        # Batched classification
        mn_classes = []
        for i in range(0, len(crops), 32):
            batch_crops = crops[i:i+32]
            texts = []
            for crop in batch_crops:
                msgs = [{"role":"user","content":[{"type":"image","image":crop},{"type":"text","text":"classify"}]}]
                texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            inputs = processor(images=batch_crops, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(**inputs, output_hidden_states=True)
                logits = cls_head(out.last_hidden_state)
                mn_classes.extend(logits.argmax(-1).cpu().tolist())

        for j in range(len(fb)):
            x1,y1,x2,y2 = fb[j]
            results.append({
                "image_id": img_path.name,
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "category_id": int(mn_classes[j]),
                "score": float(fs[j]),
            })

    with open(output_json, "w") as f:
        json.dump(results, f)
    return len(results)


def compute_map(pred_json, val_imgs, val_labs):
    """Compute detection + classification mAP@0.5."""
    from PIL import Image as PILImage

    preds_raw = json.load(open(pred_json))
    preds_by_img = defaultdict(list)
    for p in preds_raw:
        x,y,w,h = p["bbox"]
        preds_by_img[p["image_id"]].append({
            "cat": p["category_id"], "score": p["score"],
            "bbox": [x, y, x+w, y+h]})

    gt_by_img = {}; cats = set()
    for img_p in sorted(val_imgs.glob("*.jpg")):
        lab = val_labs / img_p.with_suffix(".txt").name
        if not lab.exists(): continue
        img = PILImage.open(img_p); w,h = img.size
        anns = []
        for line in lab.read_text().splitlines():
            if not line.strip(): continue
            parts = line.split(); cid = int(float(parts[0]))
            cx,cy,bw,bh = float(parts[1])*w,float(parts[2])*h,float(parts[3])*w,float(parts[4])*h
            anns.append({"cat":cid,"bbox":[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]})
            cats.add(cid)
        gt_by_img[img_p.name] = anns

    def iou(a,b):
        x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
        inter=max(0,x2-x1)*max(0,y2-y1)
        u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/u if u>0 else 0

    def voc_ap(rec,prec):
        mrec=[0]+rec+[1];mpre=[0]+prec+[0]
        for i in range(len(mpre)-2,-1,-1):mpre[i]=max(mpre[i],mpre[i+1])
        return sum((mrec[i]-mrec[i-1])*mpre[i] for i in range(1,len(mrec)) if mrec[i]!=mrec[i-1])

    def eval_map(preds,gt,cats,class_aware):
        positives=defaultdict(int)
        idx={}
        for img,anns in gt.items():
            idx[img]=[{"cat":a["cat"],"bbox":a["bbox"],"used":False} for a in anns]
            for a in anns: positives[a["cat"]]+=1
        aps={}
        for c in sorted(cats):
            dets=[]
            for img,ps in preds.items():
                for p in ps:
                    if class_aware and p["cat"]!=c: continue
                    dets.append((img,p))
            dets.sort(key=lambda x:-x[1]["score"])
            tp,fp=[],[]
            for img,p in dets:
                best_i,best_o=-1,0
                for j,g in enumerate(idx.get(img,[])):
                    if g["used"]:continue
                    if class_aware and g["cat"]!=c:continue
                    o=iou(p["bbox"],g["bbox"])
                    if o>best_o:best_o,best_i=o,j
                if best_i>=0 and best_o>=0.5:
                    idx[img][best_i]["used"]=True;tp.append(1);fp.append(0)
                else:tp.append(0);fp.append(1)
            tot=positives[c]
            if tot==0 or not dets:aps[c]=0;continue
            ct=cf=0;recs,precs=[],[]
            for t,f in zip(tp,fp):
                ct+=t;cf+=f;recs.append(ct/tot);precs.append(ct/(ct+cf))
            aps[c]=voc_ap(recs,precs)
        for img in idx:
            for g in idx[img]:g["used"]=False
        return sum(aps.values())/max(len(aps),1)

    det = eval_map(preds_by_img, gt_by_img, cats, False)
    cls = eval_map(preds_by_img, gt_by_img, cats, True)
    return det, cls, 0.7*det + 0.3*cls


def process_checkpoint(ckpt_path):
    """Full pipeline for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {ckpt_path}")
    print(f"{'='*60}")

    # Step 1: NF4 quantize
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    nf4_name = f"nf4_{ckpt_path.parent.name}.pt"
    nf4_path = EXPORT_DIR / nf4_name
    size = nf4_quantize_checkpoint(ckpt_path, nf4_path)

    # Step 2: Run inference on val set
    pred_json = Path(f"/tmp/overnight_preds_{ckpt_path.parent.name}.json")
    print(f"  Running inference on {VAL_IMGS}...")
    t0 = time.time()
    n_dets = run_inference(nf4_path, VAL_IMGS, pred_json)
    t1 = time.time()
    print(f"  {n_dets} detections in {t1-t0:.1f}s")

    # Step 3: Compute mAP
    det_map, cls_map, combined = compute_map(pred_json, VAL_IMGS, VAL_LABS)

    # Step 4: Compare to baseline
    cls_delta = cls_map - BASELINE_CLS_MAP
    combined_delta = combined - BASELINE_COMBINED

    print(f"\n  === RESULTS ===")
    print(f"  Detection mAP@0.5:      {det_map:.4f} ({det_map*100:.1f}%)")
    print(f"  Classification mAP@0.5: {cls_map:.4f} ({cls_map*100:.1f}%) [DINOv2: {BASELINE_CLS_MAP*100:.1f}%, delta: {cls_delta*100:+.1f}%]")
    print(f"  Combined:               {combined:.4f} ({combined*100:.1f}%) [DINOv2: {BASELINE_COMBINED*100:.1f}%, delta: {combined_delta*100:+.1f}%]")
    print(f"  NF4 size: {size:.0f} MB | Inference: {t1-t0:.1f}s")

    beats_baseline = combined > BASELINE_COMBINED

    if beats_baseline:
        print(f"  >>> BEATS BASELINE! <<<")
    else:
        print(f"  Below baseline by {combined_delta*100:.1f}%")

    result = {
        "checkpoint": str(ckpt_path),
        "nf4_path": str(nf4_path),
        "nf4_size_mb": size,
        "det_map": det_map, "cls_map": cls_map, "combined": combined,
        "cls_delta": cls_delta, "combined_delta": combined_delta,
        "beats_baseline": beats_baseline,
        "inference_time": t1-t0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Append to results log
    results_log = []
    if RESULTS_LOG.exists():
        results_log = json.load(open(RESULTS_LOG))
    results_log.append(result)
    with open(RESULTS_LOG, "w") as f:
        json.dump(results_log, f, indent=2)

    return result


def find_new_checkpoints(seen):
    """Find checkpoint files we haven't processed yet."""
    new = []
    for watch_dir in WATCH_DIRS:
        if not watch_dir.exists():
            continue
        # Check best/best.pt
        best = watch_dir / "best" / "best.pt"
        if best.exists() and str(best) not in seen:
            new.append(best)
        # Check checkpoint-N/checkpoint.pt
        for d in sorted(watch_dir.glob("checkpoint-*")):
            ckpt = d / "checkpoint.pt"
            if ckpt.exists() and str(ckpt) not in seen:
                new.append(ckpt)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="Process single checkpoint")
    parser.add_argument("--watch", action="store_true", help="Watch for new checkpoints continuously")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.checkpoint:
        process_checkpoint(args.checkpoint)
        return

    if args.watch:
        print("=== OVERNIGHT PIPELINE: Watching for checkpoints ===")
        print(f"Watch dirs: {[str(d) for d in WATCH_DIRS]}")
        print(f"Baseline: cls_mAP={BASELINE_CLS_MAP:.4f}, combined={BASELINE_COMBINED:.4f}")
        print(f"Interval: {args.interval}s\n")

        seen = set()
        best_combined = 0

        while True:
            new = find_new_checkpoints(seen)
            for ckpt in new:
                result = process_checkpoint(ckpt)
                seen.add(str(ckpt))
                if result["combined"] > best_combined:
                    best_combined = result["combined"]
                    print(f"\n*** NEW BEST: {best_combined:.4f} ({best_combined*100:.1f}%) ***\n")

            time.sleep(args.interval)
    else:
        # One-shot: process all existing checkpoints
        seen = set()
        checkpoints = find_new_checkpoints(seen)
        if not checkpoints:
            print("No checkpoints found. Use --watch to monitor, or --checkpoint PATH.")
            return
        for ckpt in checkpoints:
            process_checkpoint(ckpt)


if __name__ == "__main__":
    main()
