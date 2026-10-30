"""Shared lifecycle every rendering engine implements, plus the run loop itself.

The loop/timing/frame-budget mechanism is identical across all three engine
tiers on purpose (Lecture 07: the render loop is the general, engine-agnostic
principle; the specific drawing API is not). `run()` is deliberately not
overridable, so subclasses can't accidentally diverge from that shared
contract while filling in engine-specific behaviour.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

class BaseEngine(ABC):
    def __init__(self, width: int, height: int, title: str, target_fps: int = 60) -> None:
        self.width: int = width
        self.height: int = height
        self.title: str = title
        self.target_fps: int = target_fps
        self.delta_time: float = 0.0
        self.time_scale: float = 1.0
        self.vsync: bool = False
        self.resizable: bool = True
        self.clear_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
        self.time: float = 0.0

    @abstractmethod
    def setup(self) -> None:
        """Create the window/rendering context. Called once, before the loop starts."""

    @abstractmethod
    def handle_events(self) -> bool:
        """Poll input/window events for this frame.

        Return False to end the run loop (window close, Esc key, etc).
        """
    @abstractmethod
    def start(self) -> None:
        """Called once, after setup and before the first frame of the loop."""

    @abstractmethod
    def update(self, audio_features: dict[str, float|str]) -> None:
        """Advance state for this frame using the latest audio feature values."""

    @abstractmethod
    def render(self) -> None:
        """Draw the current frame."""

    @abstractmethod
    def teardown(self) -> None:
        """Release engine resources on exit (close window/context)."""

    def run(self, feature_source: Callable[[float], dict[str, float|str]]) -> None:
        """Owns the engine lifecycle: setup once, loop until handle_events()
        returns False, then teardown. Do not override this method — every
        engine tier shares this exact loop.
        """
        frame_duration = 0.0
        if self.target_fps <= 0:
            frame_duration = 0.0
        else:
            frame_duration = 1.0 / self.target_fps

        self.setup()
        self.start()
        try:
            running = True
            previous_frame_start = time.perf_counter()
            while running:
                frame_start = time.perf_counter()
                self.delta_time = (frame_start - previous_frame_start) * self.time_scale
                self.time += self.delta_time
                previous_frame_start = frame_start

                running = self.handle_events()
                if not running:
                    break

                audio_features = feature_source(self.delta_time)
                self.update(audio_features)
                self.render()

                elapsed = time.perf_counter() - frame_start
                remaining = frame_duration - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.teardown()
