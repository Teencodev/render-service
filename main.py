"""
Pippit Watermark Remover — FastAPI microservice
Deploy này lên Render.com (free tier)

Flow:
  POST /remove-watermark  { "video_url": "...", "position": "bottom-right" }
  → Download video
  → ffmpeg delogo filter
  → Upload lên file.io (tạm) hoặc trả base64
  → Return download URL
"""

import os
import uuid
import subprocess
import tempfile
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import imageio_ffmpeg

app = FastAPI(title="Pippit Watermark Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cấu hình watermark Pippit ──────────────────────────────────────────────
# Watermark Pippit nằm ở GÓC TRÊN TRÁI (logo + tên app)
# x=10, y=10 : cách mép trái/trên 10px
# w=180, h=60 : đủ rộng bao logo + chữ "Pippit"
# Nếu vẫn còn sót, tăng w hoặc h thêm 20-30px
WATERMARK_PRESETS = {
    "top-left":     {"x": "10",     "y": "10",     "w": "180", "h": "60"},  # ← DEFAULT
    "top-right":    {"x": "iw-190", "y": "10",     "w": "180", "h": "60"},
    "bottom-left":  {"x": "10",     "y": "ih-70",  "w": "180", "h": "60"},
    "bottom-right": {"x": "iw-190", "y": "ih-70",  "w": "180", "h": "60"},
}

# Thư mục lưu file tạm trên server
TEMP_DIR = tempfile.gettempdir()

# ── Models ─────────────────────────────────────────────────────────────────
class RemoveRequest(BaseModel):
    video_url: str
    position: Optional[str] = "top-left"
    # Override tọa độ thủ công nếu muốn
    x: Optional[str] = None
    y: Optional[str] = None
    w: Optional[str] = None
    h: Optional[str] = None

class RemoveResponse(BaseModel):
    success: bool
    download_url: Optional[str] = None
    file_id: Optional[str] = None
    error: Optional[str] = None

# ── Storage tạm trong memory (file_id → local path) ───────────────────────
_file_store: dict[str, str] = {}

# ── Helper: download video ─────────────────────────────────────────────────
async def download_video(url: str, dest: str) -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes(8192):
                    f.write(chunk)

# ── Helper: run ffmpeg ─────────────────────────────────────────────────────
def run_ffmpeg(input_path: str, output_path: str, delogo: dict) -> None:
    """
    Dùng ffmpeg delogo filter để xóa/blur watermark.
    delogo = {"x": ..., "y": ..., "w": ..., "h": ...}
    """
    x, y, w, h = delogo["x"], delogo["y"], delogo["w"], delogo["h"]
    vf = f"delogo=x={x}:y={y}:w={w}:h={h}"

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")

# ── Background cleanup ─────────────────────────────────────────────────────
async def cleanup_after_delay(file_id: str, path: str, delay: int = 300):
    """Xóa file sau 5 phút"""
    await asyncio.sleep(delay)
    try:
        os.remove(path)
        _file_store.pop(file_id, None)
    except:
        pass

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Pippit Watermark Remover"}

@app.get("/health")
def health():
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, timeout=5)
        ffmpeg_ok = result.returncode == 0
    except:
        ffmpeg_ok = False
    return {"status": "ok", "ffmpeg": ffmpeg_ok}

@app.post("/remove-watermark", response_model=RemoveResponse)
async def remove_watermark(req: RemoveRequest, background_tasks: BackgroundTasks):
    file_id  = str(uuid.uuid4())
    in_path  = os.path.join(TEMP_DIR, f"{file_id}_input.mp4")
    out_path = os.path.join(TEMP_DIR, f"{file_id}_output.mp4")

    try:
        # 1. Download video
        await download_video(req.video_url, in_path)

        # 2. Xác định tọa độ watermark
        if req.x and req.y and req.w and req.h:
            # Tọa độ tùy chỉnh từ client
            delogo = {"x": req.x, "y": req.y, "w": req.w, "h": req.h}
        else:
            preset = req.position or "bottom-right"
            if preset not in WATERMARK_PRESETS:
                raise HTTPException(400, f"Invalid position. Choose: {list(WATERMARK_PRESETS.keys())}")
            delogo = WATERMARK_PRESETS[preset]

        # 3. Chạy ffmpeg
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_ffmpeg, in_path, out_path, delogo)

        # 4. Lưu output vào store
        _file_store[file_id] = out_path
        background_tasks.add_task(cleanup_after_delay, file_id, out_path, 300)

        # Xóa input
        try: os.remove(in_path)
        except: pass

        return RemoveResponse(
            success=True,
            download_url=f"/download/{file_id}",
            file_id=file_id,
        )

    except httpx.HTTPError as e:
        return RemoveResponse(success=False, error=f"Download failed: {e}")
    except RuntimeError as e:
        return RemoveResponse(success=False, error=str(e))
    except Exception as e:
        return RemoveResponse(success=False, error=f"Unexpected error: {e}")
    finally:
        try: os.remove(in_path)
        except: pass

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    from fastapi.responses import FileResponse
    path = _file_store.get(file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found or expired (5 phút TTL)")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="video_no_watermark.mp4",
        headers={"Content-Disposition": "attachment; filename=video_no_watermark.mp4"}
    )
