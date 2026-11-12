import math
import moderngl
import pyglet
import moderngl_window as mglw
from scipy.spatial.transform import Rotation
from moderngl_window.conf import settings
from moderngl_window.context.base.window import BaseWindow
from engines.base_engine import BaseEngine
from engines.mesh import Mesh
from engines.utils import Utils

class ModernGLEngine(BaseEngine):
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

    def _on_key_event(self, key: int, action: int, modifiers: int) -> None:
        if key == self.window.keys.ESCAPE and action == self.window.keys.ACTION_PRESS:
            self._should_close = True

    def handle_events(self) -> bool:
        return not (self.window.is_closing or self._should_close)

    def start(self) -> None:
        shader_program = Utils.create_standard_shader(self.ctx)
        Utils.set_shader_texture(shader_program, Utils.generate_default_texture(self.ctx, "checkerboard"), "maintex")
        self.cube_mesh: Mesh = Utils.create_cube_mesh(shader_program)
        self.cube_mesh.position = [0.0, 0.0, -2.5]
        self.lerped_features["amplitude"] = 1.0

    def update(self, audio_features: dict[str, float|str]) -> None:
        delta_time: float = self.get_delta_time()

        raw_amplitude = audio_features.get("amplitude", 1.0)
        amplitude: float = (float(raw_amplitude) * 2.0) + 0.5

        self.lerped_features["amplitude"] = Utils.lerp(self.lerped_features["amplitude"], amplitude, 15 * delta_time)
        self.cube_mesh.scale = [self.lerped_features["amplitude"], self.lerped_features["amplitude"], self.lerped_features["amplitude"]]

        self.cube_mesh.rotation = Rotation.from_euler("xyz", [self.time, self.time, self.time])

        print(f"FPS: {1.0 / delta_time if delta_time > 0 else 0.0}")

    def render(self) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.window.clear(*self.clear_colour) # type: ignore
        view_projection_matrix = Utils.get_projection_matrix(math.radians(90.0), self.width / self.height)
        self.cube_mesh.draw(view_projection_matrix)
        self.window.swap_buffers()

    def teardown(self) -> None:
        self.window.close()
