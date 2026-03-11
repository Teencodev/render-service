"""
Pippit Watermark Remover — FastAPI async job queue
Flow mới (không bị PHP timeout):
  1. POST /submit-job        → trả job_id ngay lập tức
  2. GET  /job/{job_id}      → polling status: pending/downloading/processing/done/failed
  3. GET  /download/{job_id} → tải file khi status=done
"""

import os, uuid, subprocess, tempfile, asyncio, imageio_ffmpeg, httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Pippit Watermark Remover")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WATERMARK_PRESETS = {
    "top-left":     {"x": "10",     "y": "10",    "w": "180", "h": "60"},
    "top-right":    {"x": "iw-190", "y": "10",    "w": "180", "h": "60"},
    "bottom-left":  {"x": "10",     "y": "ih-70", "w": "180", "h": "60"},
    "bottom-right": {"x": "iw-190", "y": "ih-70", "w": "180", "h": "60"},
}

TEMP_DIR = tempfile.gettempdir()

# job_id → {status, download_url, error, out_path}
_jobs: dict[str, dict] = {}

class JobRequest(BaseModel):
    video_url: str
    position: Optional[str] = "top-left"
    x: Optional[str] = None
    y: Optional[str] = None
    w: Optional[str] = None
    h: Optional[str] = None

async def download_video(url: str, dest: str) -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes(8192):
                    f.write(chunk)

def run_ffmpeg(input_path: str, output_path: str, delogo: dict) -> None:
    x, y, w, h = delogo["x"], delogo["y"], delogo["w"], delogo["h"]
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "copy",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

async def cleanup_later(job_id: str, path: str, delay: int = 600):
    await asyncio.sleep(delay)
    try: os.remove(path)
    except: pass
    _jobs.pop(job_id, None)

async def process_job(job_id: str, req: JobRequest):
    in_path  = os.path.join(TEMP_DIR, f"{job_id}_input.mp4")
    out_path = os.path.join(TEMP_DIR, f"{job_id}_output.mp4")
    try:
        _jobs[job_id]["status"] = "downloading"
        await download_video(req.video_url, in_path)
        size = os.path.getsize(in_path)
        if size < 1000:
            raise RuntimeError(f"Video too small: {size} bytes")
        _jobs[job_id]["status"] = "processing"
        if req.x and req.y and req.w and req.h:
            delogo = {"x": req.x, "y": req.y, "w": req.w, "h": req.h}
        else:
            delogo = WATERMARK_PRESETS.get(req.position or "top-left", WATERMARK_PRESETS["top-left"])
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_ffmpeg, in_path, out_path, delogo)
        _jobs[job_id]["status"]       = "done"
        _jobs[job_id]["out_path"]     = out_path
        _jobs[job_id]["download_url"] = f"/download/{job_id}"
        asyncio.create_task(cleanup_later(job_id, out_path, 600))
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = str(e)
    finally:
        try: os.remove(in_path)
        except: pass

@app.get("/")
def root():
    return {"status": "ok", "service": "Pippit Watermark Remover v2"}

@app.get("/health")
def health():
    try:
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], capture_output=True, timeout=5)
        ffmpeg_ok = r.returncode == 0
    except:
        ffmpeg_ok = False
    return {"status": "ok", "ffmpeg": ffmpeg_ok}

@app.post("/submit-job")
async def submit_job(req: JobRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "download_url": None, "error": None, "out_path": None}
    background_tasks.add_task(process_job, job_id, req)
    return {"job_id": job_id, "status": "pending"}

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
