"""
Pippit Watermark Remover — Upload video → xóa watermark → download
"""

import os, uuid, asyncio, subprocess, tempfile, shutil
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Pippit Watermark Remover")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMP_DIR = tempfile.gettempdir()
_jobs: dict[str, dict] = {}

# ── Auto-detect watermark ─────────────────────────────────────────────────────
def auto_detect_watermark(frames, mean_frame, width, height):
    stack    = np.stack(frames, axis=0)
    std_map  = np.std(stack, axis=0).mean(axis=2)
    corner_h = max(60,  int(height * 0.10))
    corner_w = max(140, int(width  * 0.15))
    corners  = [
        (0,               0,               corner_h,  corner_w),
        (0,               width-corner_w,  corner_h,  width),
        (height-corner_h, 0,               height,    corner_w),
        (height-corner_h, width-corner_w,  height,    width),
    ]
    best, best_score = None, 0
    for r1, c1, r2, c2 in corners:
        roi_gray     = cv2.cvtColor(mean_frame[r1:r2, c1:c2], cv2.COLOR_BGR2GRAY)
        edges        = cv2.Canny(roi_gray, 20, 60)
        edge_density = edges.mean() / 255.0
        stability    = 1.0 / (1.0 + std_map[r1:r2, c1:c2].mean())
        score        = edge_density * stability
        ys, xs       = np.where(edges > 0)
        if score > best_score and edge_density > 0.002 and len(xs) > 20:
            best_score = score
            pad = 10
            x = max(0, c1 + int(xs.min()) - pad)
            y = max(0, r1 + int(ys.min()) - pad)
            w = min(width  - x, int(xs.max() - xs.min()) + 1 + 2*pad)
            h = min(height - y, int(ys.max() - ys.min()) + 1 + 2*pad)
            best = (x, y, w, h)
    return best

def build_mask(mean_frame, region_xywh, frame_shape):
    x, y, w, h = region_xywh
    H, W       = frame_shape[:2]
    roi_gray   = cv2.cvtColor(mean_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    edges      = cv2.Canny(roi_gray, 30, 80)
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated    = cv2.dilate(edges, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    clean      = np.zeros_like(dilated)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 80:
            clean[labels == i] = 255
    if clean.sum() == 0:
        clean = np.full((h, w), 255, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y:y+h, x:x+w] = clean
    return mask

def remove_watermark_opencv(input_path: str, output_path: str, manual_region=None):
    cap    = cv2.VideoCapture(input_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample ~60 frames
    sample_frames, step = [], max(1, total // 60)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret: sample_frames.append(f.astype(np.float32))
        if len(sample_frames) >= 60: break

    if not sample_frames:
        cap.release()
        raise RuntimeError("Cannot read frames from video")

    mean_frame = np.mean(np.stack(sample_frames), axis=0).astype(np.uint8)
    region     = manual_region or auto_detect_watermark(sample_frames, mean_frame, width, height)
    if region is None:
        cap.release()
        raise RuntimeError("Auto-detection failed. No watermark found.")

    mask       = build_mask(mean_frame, region, (height, width))
    frames_dir = tempfile.mkdtemp(prefix="pippit_wm_")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            result = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.png"), result)
        cap.release()

        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except:
            ffmpeg_bin = "ffmpeg"

        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "%06d.png"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

async def cleanup_later(job_id: str, path: str, delay: int = 600):
    await asyncio.sleep(delay)
    try: os.remove(path)
    except: pass
    _jobs.pop(job_id, None)

async def process_job(job_id: str, input_path: str, manual_region=None):
    out_path = os.path.join(TEMP_DIR, f"{job_id}_output.mp4")
    try:
        _jobs[job_id]["status"] = "processing"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, remove_watermark_opencv, input_path, out_path, manual_region)
        _jobs[job_id]["status"]       = "done"
        _jobs[job_id]["out_path"]     = out_path
        _jobs[job_id]["download_url"] = f"/download/{job_id}"
        asyncio.create_task(cleanup_later(job_id, out_path, 600))
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = str(e)
    finally:
        try: os.remove(input_path)
        except: pass

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Pippit Watermark Remover v4 (upload)"}

@app.get("/health")
def health():
    try: cv2.Canny(np.zeros((10,10), np.uint8), 50, 100); cv2_ok = True
    except: cv2_ok = False
    return {"status": "ok", "opencv": cv2_ok}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload video → nhận job_id ngay → polling /job/{id}"""
    job_id   = str(uuid.uuid4())
    in_path  = os.path.join(TEMP_DIR, f"{job_id}_input.mp4")

    # Lưu file upload vào disk
    with open(in_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    size = os.path.getsize(in_path)
    if size < 1000:
        os.remove(in_path)
        raise HTTPException(400, "File quá nhỏ hoặc rỗng")

    _jobs[job_id] = {"status": "pending", "download_url": None, "error": None, "out_path": None}
    background_tasks.add_task(process_job, job_id, in_path)
    return {"job_id": job_id, "status": "pending", "size_bytes": size}

@app.get("/job/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found or expired")
    return {
        "job_id":       job_id,
        "status":       job["status"],
        "download_url": job.get("download_url"),
        "error":        job.get("error"),
    }

@app.get("/download/{job_id}")
def download_file(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "File not ready or expired")
    path = job.get("out_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found on disk")
    return FileResponse(
        path, media_type="video/mp4", filename="video_no_watermark.mp4",
        headers={"Content-Disposition": "attachment; filename=video_no_watermark.mp4"}
    )
