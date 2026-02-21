from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
        self.lock = threading.Lock()

    def start_async(self, config: PipelineConfig):
        with self.lock:
            if self.state != PipelineState.STOPPED:
                raise RuntimeError("Pipeline already running")

            self.state = PipelineState.STARTING
            self.pipeline = RTSPYoloPipeline(**config.dict())

            threading.Thread(target=self._background_start).start()

    def _background_start(self):
        try:
            self.pipeline.start()
            self.state = PipelineState.RUNNING
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

    def status(self):
        return {
            "state": self.state,
            "error": self.error
        }
