"""
NM i AI 2026 - Local Eval Server

Accepts ZIP file uploads, runs them through the exact sandbox replica,
and returns pass/fail + timing analysis.

Usage:
    python eval_server.py [--port 8765] [--images /path/to/images]

Endpoints:
    POST /eval     Upload a submission.zip, runs full sandbox pipeline
    GET  /status   Check if server is ready / busy
    GET  /health   Health check

The server runs each submission inside Docker with competition constraints:
  - Python 3.11, 4 vCPU, 8GB RAM, L4 GPU, no network, 300s timeout

Requires: Docker with nvidia runtime, the nmiai-sandbox image built.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

DEFAULT_IMAGES = str(
    pathlib.Path(__file__).parent.parent
    / "data-creation/data/coco_dataset/train/images"
)
DOCKER_IMAGE = "nmiai-sandbox"
TIMEOUT = 360  # 300s inference + 60s model loading grace period

# Server state
_busy = False
_last_result = None


class EvalHandler(BaseHTTPRequestHandler):
    images_dir = DEFAULT_IMAGES
    annotations_path = ""

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path == "/status":
            self._json_response(200, {
                "busy": _busy,
                "last_result": _last_result,
            })
        else:
            self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        global _busy, _last_result

        if self.path != "/eval":
            self._json_response(404, {"error": "Not found"})
            return

        if _busy:
            self._json_response(429, {"error": "Server busy, try again later"})
            return

        # Read uploaded ZIP
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._json_response(400, {"error": "No data received"})
            return

        if content_length > 450 * 1024 * 1024:  # 450MB safety margin
            self._json_response(400, {"error": f"File too large: {content_length / 1024 / 1024:.0f}MB"})
            return

        _busy = True
        tmpdir = None
        try:
            # Save ZIP to temp file
            tmpdir = tempfile.mkdtemp(prefix="nmiai_eval_")
            zip_path = pathlib.Path(tmpdir) / "submission.zip"
            with open(zip_path, "wb") as f:
                remaining = content_length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)

            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"[EVAL] Received {zip_size_mb:.1f}MB ZIP, running sandbox...")

            # Count images
            images_dir = pathlib.Path(self.images_dir)
            image_count = len(
                list(images_dir.glob("*.jpg"))
                + list(images_dir.glob("*.jpeg"))
                + list(images_dir.glob("*.png"))
            )

            # Check GPU
            gpu_flag = ""
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                gpu_flag = "--gpus all"
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Run Docker container with competition constraints
            start = time.time()
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{zip_path}:/submission.zip:ro",
                "-v", f"{self.images_dir}:/data/images:ro",
                "--memory=8g",
                "--memory-swap=8g",
                "--cpus=4",
                "--network=none",
                "--pids-limit=256",
            ]
            # Mount annotations for scoring
            annotations_path = pathlib.Path(self.annotations_path)
            if annotations_path.exists():
                cmd.extend(["-v", f"{annotations_path}:/data/annotations.json:ro"])
            if gpu_flag:
                cmd.extend(["--gpus", "all"])
            cmd.extend([
                DOCKER_IMAGE,
                "--zip", "/submission.zip",
                "--input", "/data/images",
                "--output", "/predictions.json",
                "--annotations", "/data/annotations.json",
                "--timeout", str(TIMEOUT),
            ])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT + 30,  # extra buffer for container overhead
            )
            elapsed = time.time() - start

            response = {
                "passed": result.returncode == 0,
                "exit_code": result.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "timeout_seconds": TIMEOUT,
                "budget_used_pct": round(elapsed / TIMEOUT * 100, 1),
                "zip_size_mb": round(zip_size_mb, 1),
                "image_count": image_count,
                "per_image_seconds": round(elapsed / max(image_count, 1), 2),
                "stdout": result.stdout[-3000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }

            # Predict timeout risk
            if image_count > 0:
                per_img = elapsed / image_count
                for n in [100, 150, 200, 250, 300]:
                    response[f"est_{n}_images"] = {
                        "seconds": round(per_img * n, 1),
                        "would_pass": per_img * n < TIMEOUT,
                    }

            _last_result = response
            status = 200 if result.returncode == 0 else 422
            print(f"[EVAL] {'PASSED' if result.returncode == 0 else 'FAILED'} in {elapsed:.1f}s")
            self._json_response(status, response)

        except subprocess.TimeoutExpired:
            _last_result = {"passed": False, "error": "Container timed out"}
            self._json_response(422, _last_result)
        except Exception as e:
            _last_result = {"passed": False, "error": str(e)}
            self._json_response(500, _last_result)
        finally:
            _busy = False
            if tmpdir and pathlib.Path(tmpdir).exists():
                shutil.rmtree(tmpdir, ignore_errors=True)

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        if "/health" not in str(args):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 Local Eval Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--images", default=DEFAULT_IMAGES, help="Path to test images")
    parser.add_argument("--annotations", default="", help="Path to COCO annotations.json for scoring")
    args = parser.parse_args()

    # Auto-detect annotations next to images dir
    if not args.annotations:
        auto_ann = pathlib.Path(args.images).parent / "annotations.json"
        if auto_ann.exists():
            args.annotations = str(auto_ann)
            print(f"Auto-detected annotations: {auto_ann}")

    # Validate
    images_dir = pathlib.Path(args.images)
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        exit(1)

    image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")))
    print(f"Images: {images_dir} ({image_count} images)")

    # Check Docker image exists
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Docker image '{DOCKER_IMAGE}' not found!")
        print(f"Build it: docker build -t {DOCKER_IMAGE} {pathlib.Path(__file__).parent}")
        exit(1)

    EvalHandler.images_dir = str(images_dir)
    EvalHandler.annotations_path = args.annotations
    if args.annotations:
        print(f"Annotations: {args.annotations} (scoring enabled)")
    else:
        print("Annotations: not provided (scoring disabled)")

    server = HTTPServer(("0.0.0.0", args.port), EvalHandler)
    print(f"\nEval server running on http://0.0.0.0:{args.port}")
    print(f"  POST /eval   - Upload submission.zip for evaluation")
    print(f"  GET  /status  - Check server status")
    print(f"  GET  /health  - Health check")
    print(f"\nUsage: curl -X POST -F 'file=@submission.zip' http://localhost:{args.port}/eval")
    print(f"   or: curl -X POST --data-binary @submission.zip http://localhost:{args.port}/eval")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
