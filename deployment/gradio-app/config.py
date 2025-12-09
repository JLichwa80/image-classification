from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class AppConfig:
    app_dir: Path
    content_dir: Path
    examples_dir: Path
    counter_file: Path

    stage1_model_path: str = 'models/set2_pneumonia_detector.pkl'
    stage2_model_path: str = 'models/set2_stage2_bacterial_viral_detector.pkl'

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    clahe_colormap: str = 'HOT'

    stage_1_key: str = 'Stage 1 - Pneumonia Detection'
    stage_2_key: str = 'Stage 2 - Pneumonia Type'

    @property
    def clahe_settings(self) -> Dict:
        return {
            "clipLimit": self.clahe_clip_limit,
            "tileGridSize": self.clahe_tile_grid_size,
            "colormap": self.clahe_colormap
        }

    @property
    def classification_icons(self) -> Dict[str, Path]:
        icon_dir = self.content_dir / "images"
        return {
            'normal': icon_dir / 'normal.png',
            'bacterial': icon_dir / 'bacterial.png',
            'viral': icon_dir / 'viral.png'
        }

    @classmethod
    def from_app_dir(cls, app_dir: Path) -> 'AppConfig':
        return cls(
            app_dir=app_dir,
            content_dir=app_dir / "content",
            examples_dir=app_dir / "examples",
            counter_file=(app_dir / "data" / "total_images_processed.txt").resolve()
        )
