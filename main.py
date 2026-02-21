import subprocess
import threading
import queue
import time
import signal
import sys
import enum
from typing import Optional

import numpy as np
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =====================================================
# ===================== PIPELINE ======================
# =====================================================

class RTSPYoloPipeline:
    def __init__(
        self,
        rtsp_in: str,
        rtsp_out: str,
        model_path: str,
        conf: float,
        iou: float,
        width: int,
        height: int,
        fps: int,
        queue_size: int = 5,
    ):
        self.rtsp_in = rtsp_in
        self.rtsp_out = rtsp_out
        self.width = width
        self.height = height
        self.fps = fps
        self.conf = conf
        self.iou = iou

        self.frame_size = width * height * 3
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False

        self.model = YOLO(model_path)
        self.model.to("cuda")

        self.reader_thread = None
        self.infer_thread = None

    def start(self):
        self.running = True
        self.reader_thread = threading.Thread(target=self._rtsp_reader, daemon=True)
        self.infer_thread = threading.Thread(target=self._infer_and_stream, daemon=True)
        self.reader_thread.start()
        self.infer_thread.start()

    def stop(self):
        self.running = False

    # ---------------- RTSP IN ----------------
    def _rtsp_reader(self):
        while self.running:
            try:
                self._run_rtsp_reader()
            except Exception:
                time.sleep(1)

    def _run_rtsp_reader(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.rtsp_in,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

        while self.running:
            raw = proc.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                proc.kill()
                raise RuntimeError("RTSP input broken")

            frame = np.frombuffer(raw, np.uint8).reshape(
                (self.height, self.width, 3)
            )

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(frame)

    # ---------------- INFERENCE + RTSP OUT ----------------
    def _infer_and_stream(self):
        while self.running:
            try:
                self._run_infer_loop()
            except Exception:
                time.sleep(1)

    def _run_infer_loop(self):
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "rtsp",
            self.rtsp_out
        ]

        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        last_time = time.time()
        fps_ema = 0.0

        while self.running:
            frame = self.queue.get()

            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            annotated = results[0].plot()

            now = time.time()
            inst_fps = 1 / max(now - last_time, 1e-6)
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            last_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps_ema:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            ffmpeg_proc.stdin.write(annotated.tobytes())


# =====================================================
# ===================== CONTROLLER ====================
# =====================================================

class PipelineState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class PipelineConfig(BaseModel):
    rtsp_in: str
    rtsp_out: str
    model_path: str = "yolov8n.pt"
    conf: float = 0.4
    iou: float = 0.5
    width: int = 1280
    height: int = 720
    fps: int = 25


class SinglePipelineController:
    def __init__(self):
        self.pipeline: Optional[RTSPYoloPipeline] = None
        self.state = PipelineState.STOPPED
        self.error: Optional[str] = None
        self.rtsp_url: Optional[str] = None
        self.lock = threading.Lock()

    def start_async(self, config: PipelineConfig):
        with self.lock:
            if self.state in (PipelineState.STARTING, PipelineState.RUNNING):
                raise RuntimeError("Pipeline already running")

            self.state = PipelineState.STARTING
            self.error = None
            self.rtsp_url = config.rtsp_out

            self.pipeline = RTSPYoloPipeline(
                rtsp_in=config.rtsp_in,
                rtsp_out=config.rtsp_out,
                model_path=config.model_path,
                conf=config.conf,
                iou=config.iou,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )

            threading.Thread(
                target=self._background_start,
                daemon=True
            ).start()

    def _background_start(self):
        try:
            self.pipeline.start()

            if self._wait_for_rtsp(self.rtsp_url):
                self.state = PipelineState.RUNNING
            else:
                raise RuntimeError("RTSP output not reachable")

        except Exception as e:
            self.state = PipelineState.ERROR
            self.error = str(e)

    def stop(self):
        with self.lock:
            if self.pipeline:
                self.pipeline.stop()
            self.pipeline = None
            self.state = PipelineState.STOPPED
            self.error = None
            self.rtsp_url = None

    def status(self):
        return {
            "state": self.state,
            "rtsp_url": self.rtsp_url,
            "error": self.error
        }

    def _wait_for_rtsp(self, url: str, timeout: int = 5) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            proc = subprocess.run(
                ["ffprobe", "-rtsp_transport", "tcp", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if proc.returncode == 0:
                return True
            time.sleep(0.5)
        return False


# =====================================================
# ===================== FASTAPI =======================
# =====================================================

app = FastAPI()
controller = SinglePipelineController()


@app.post("/pipeline/start")
def start_pipeline(config: PipelineConfig):
    try:
        controller.start_async(config)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "rtsp_url": config.rtsp_out,
        "status": PipelineState.STARTING
    }


@app.post("/pipeline/stop")
def stop_pipeline():
    controller.stop()
    return {"status": PipelineState.STOPPED}


@app.get("/pipeline/status")
def pipeline_status():
    return controller.status()


@app.on_event("shutdown")
def shutdown():
    controller.stop()
