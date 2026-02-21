import subprocess
import threading
import queue
import time
import enum
import signal
from typing import Optional

import numpy as np
import cv2
from ultralytics import YOLO

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

        self.running = threading.Event()
        self.reader_thread: Optional[threading.Thread] = None
        self.infer_thread: Optional[threading.Thread] = None

        self.reader_proc: Optional[subprocess.Popen] = None
        self.writer_proc: Optional[subprocess.Popen] = None

        self.model = YOLO(model_path)
        self.model.to("cuda")

    # ---------------- LIFECYCLE ----------------

    def start(self):
        self.running.set()

        self.reader_thread = threading.Thread(target=self._rtsp_reader)
        self.infer_thread = threading.Thread(target=self._infer_and_stream)

        self.reader_thread.start()
        self.infer_thread.start()

    def stop(self):
        self.running.clear()

        self._shutdown_ffmpeg(self.reader_proc)
        self._shutdown_ffmpeg(self.writer_proc)

        if self.reader_thread:
            self.reader_thread.join(timeout=2)
        if self.infer_thread:
            self.infer_thread.join(timeout=2)

    # ---------------- FFmpeg helpers ----------------

    def _shutdown_ffmpeg(self, proc: Optional[subprocess.Popen]):
        if not proc:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    # ---------------- RTSP INPUT ----------------

    def _rtsp_reader(self):
        while self.running.is_set():
            try:
                self._run_rtsp_reader()
            except Exception:
                time.sleep(1)

    def _run_rtsp_reader(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-stimeout", "5000000",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.rtsp_in,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        self.reader_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        while self.running.is_set():
            raw = self.reader_proc.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                raise RuntimeError("RTSP input stream broken")

            frame = np.frombuffer(raw, np.uint8).reshape(
                (self.height, self.width, 3)
            )

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(frame)

    # ---------------- INFERENCE + RTSP OUTPUT ----------------

    def _infer_and_stream(self):
        while self.running.is_set():
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

        self.writer_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        last_time = time.time()
        fps_ema = 0.0

        while self.running.is_set():
            try:
                frame = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.writer_proc.poll() is not None:
                raise RuntimeError("FFmpeg encoder exited")

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

            try:
                self.writer_proc.stdin.write(annotated.tobytes())
                self.writer_proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("FFmpeg stdin broken")
