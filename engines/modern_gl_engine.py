import math
import moderngl
import pyglet
import moderngl_window as mglw
from moderngl_window.conf import settings
from moderngl_window import BaseWindow
from engines.base_engine import BaseEngine
from engines.Mesh import Mesh
from engines.Utils import Utils

class ModernGLEngine(BaseEngine):
    """Uses moderngl-window for windowing, via its lower-level manual-loop API
    (create_window_from_settings) rather than the WindowConfig/run_window_config
    pattern shown in moderngl-window's own docs — loop ownership stays with
    BaseEngine.run() so all three engine tiers share one control-flow shape.
    """

    def setup(self) -> None:
        settings.WINDOW["class"] = "moderngl_window.context.pyglet.Window"
        settings.WINDOW["size"] = (self.width, self.height)
        settings.WINDOW["title"] = self.title
        settings.WINDOW["resizable"] = self.resizable
        settings.WINDOW["vsync"] = self.vsync

        self.window: BaseWindow = mglw.create_window_from_settings()
        self.ctx: moderngl.Context = self.window.ctx
        self.time_scaling: bool = True
        self._should_close: bool = False
        self.window.key_event_func = self._on_key_event
        self.audio_features: dict[str, float|str] = {}
        self.lerped_features: dict[str, float] = {}

    def get_delta_time(self) -> float:
        if self.time_scaling:
            return self.delta_time
        return 1.0

    def _on_key_event(self, key, action, modifiers) -> None:
        if key == self.window.keys.ESCAPE and action == self.window.keys.ACTION_PRESS:
            self._should_close = True

    def handle_events(self) -> bool:
        return not (self.window.is_closing or self._should_close)

    def start(self) -> None:
        shader_program = Utils.create_debug_shader(self.ctx)
        self.cube_mesh: Mesh = Utils.create_cube_mesh(shader_program)
        self.cube_mesh.position = (0.0, 0.0, -5.0)
        self.lerped_features["amplitude"] = 1.0

    def update(self, audio_features: dict[str, float|str]) -> None:
        delta_time: float = self.get_delta_time()

        amplitude: float = (audio_features.get("amplitude", 1.0) * 2.0) + 0.5
        self.lerped_features["amplitude"] = Utils.lerp(self.lerped_features["amplitude"], amplitude, 15 * delta_time)
        self.cube_mesh.scale = (self.lerped_features["amplitude"], self.lerped_features["amplitude"], self.lerped_features["amplitude"])
        self.cube_mesh.rotation = (self.time, self.time, self.time)

        print(f"FPS: {1.0 / delta_time if delta_time > 0 else 0.0}")

    def render(self) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.window.clear(*self.clear_color) # type: ignore
        view_projection_matrix = Utils.get_projection_matrix(math.radians(90.0), self.width / self.height, 0.1, 100.0)
        self.cube_mesh.draw(view_projection_matrix)
        self.window.swap_buffers()

    def teardown(self) -> None:
        self.window.close()
