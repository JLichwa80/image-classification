import gradio as gr
from fastai.vision.all import load_learner
import pathlib
from pathlib import Path
import platform
import logging
import sys

from custom_transforms import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, COLORMAP_SELECTION
from config import AppConfig
from view import create_app_ui
from image_processing import predict_single, predict_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


class SessionCounter:
    def __init__(self, counter_file: Path):
        self.counter_file = counter_file
        self._total = self._load()

    def _load(self) -> int:
        if not self.counter_file.exists():
            return 0
        try:
            text = self.counter_file.read_text(encoding="utf-8").strip()
            return int(text) if text else 0
        except (ValueError, IOError):
            return 0

    def _persist(self) -> None:
        self.counter_file.parent.mkdir(parents=True, exist_ok=True)
        self.counter_file.write_text(str(max(0, self._total)), encoding="utf-8")

    def increment(self, amount: int = 1) -> int:
        if amount > 0:
            self._total += amount
            self._persist()
        return self._total

    @property
    def total(self) -> int:
        return self._total

    def message(self) -> str:
        return f"Total images processed: {self._total}"


config = AppConfig.from_app_dir(Path(__file__).parent)
config.clahe_clip_limit = CLAHE_CLIP_LIMIT
config.clahe_tile_grid_size = CLAHE_TILE_GRID_SIZE
config.clahe_colormap = COLORMAP_SELECTION

counter = SessionCounter(config.counter_file)

logger.info(f"Loading Stage 1 model from {config.stage1_model_path}")
stage1_model = load_learner(config.stage1_model_path)
logger.info(f"Loading Stage 2 model from {config.stage2_model_path}")
stage2_model = load_learner(config.stage2_model_path)
logger.info("Models loaded")

gr.set_static_paths(paths=[str(config.app_dir / "assets"), str(config.examples_dir)])


def handle_single_predict(input_image):
    return predict_single(
        input_image, stage1_model, stage2_model, config.clahe_settings,
        config.classification_icons, config.stage_1_key, config.stage_2_key,
        counter.message, counter.increment
    )


def handle_batch_predict(image_files):
    return predict_batch(
        image_files, stage1_model, stage2_model, config.clahe_settings,
        config.stage_1_key, config.stage_2_key, counter.message, counter.increment
    )


demo = create_app_ui(
    config.content_dir,
    config.examples_dir,
    counter.message,
    handle_single_predict,
    handle_batch_predict
)

if __name__ == "__main__":
    import os
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    demo.launch(
        allowed_paths=[str(config.examples_dir), str(config.app_dir / "assets")],
        show_error=True,
        inbrowser=False
    )
