"""Wires the audio pipeline and rendering engine together and runs the app.

ENGINE and AUDIO_FILE are fixed, version-controlled choices, not runtime
options — committing to one engine tier and one audio file is a project
decision, not something to toggle per run.
"""

from collections.abc import Callable

from pipeline import AudioPipeline
from engines.base_engine import BaseEngine
from engines.modern_gl_engine import ModernGLEngine
import time

# Set this to whichever engine tier your project uses. See the README for
# how/why this is a one-time project decision, not a runtime toggle.
ENGINE: type[BaseEngine] = ModernGLEngine

AUDIO_FILE: str = "./audio/Rain_World_-_Threat_-_Sunken_Pier.mp3"
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_TITLE: str = "CLASS-2a Project"
TARGET_FPS: int = 0


def run_full_app() -> None:
    engine = ENGINE(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, title=WINDOW_TITLE, target_fps=TARGET_FPS)

    pipeline = AudioPipeline(file_path=AUDIO_FILE)
    try:
        pipeline.load()
    except NotImplementedError:
        print("Audio file format not supported. Skipping audio pipeline.")
        engine.run(feature_source=lambda delta_time: {})
        return

    engine.run(feature_source=lambda delta_time: pipeline.get_features(delta_time))



if __name__ == "__main__":
    run_full_app()
