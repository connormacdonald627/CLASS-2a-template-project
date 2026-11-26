import math
import random
from dataclasses import dataclass
import moderngl
import numpy as np
import moderngl_window as mglw
from scipy.spatial.transform import Rotation
from moderngl_window.conf import settings
from moderngl_window.context.base.window import BaseWindow
from engines.base_engine import BaseEngine
from engines.mesh import Mesh
from engines.utils import Utils
from engines.vector import Vector3, Vector2
from engines.texture import Texture
from engines.load_obj import LoadObj


class Galaxy:
    def __init__(self) -> None:
        self.particles: list[Galaxy.GalaxyParticle] = []
        self.arms: int = 3
        self.radius: float = 100.0
        self.distribution: Vector2 = Vector2(0.15, 0.3)
        self.speed: float = .25
    class GalaxyParticle:
        def __init__(self, position: Vector3, velocity: Vector3, scale: float = 1.0, colour: Vector3 = Vector3(1.0, 1.0, 1.0)) -> None:
            self.position = position
            self.velocity = velocity
            self.scale = scale
            self.colour: Vector3 = colour


class ModernGLEngine(BaseEngine):
    def setup(self) -> None:
        settings.WINDOW["class"] = "moderngl_window.context.pyglet.Window"
        settings.WINDOW["size"] = (self.width, self.height)
        settings.WINDOW["title"] = self.title
        settings.WINDOW["resizable"] = self.resizable
        settings.WINDOW["vsync"] = self.vsync

        self.window: BaseWindow = mglw.create_window_from_settings()
        self.ctx: moderngl.Context = self.window.ctx
        self._should_close: bool = False
        self.window.key_event_func = self._on_key_event
        self.audio_features: dict[str, float | str] = {}
        self.lerped_features: dict[str, float] = {}
        self.galaxy: Galaxy = Galaxy()

    def _on_key_event(self, key: int, action: int, modifiers: int) -> None:
        if key == self.window.keys.ESCAPE and action == self.window.keys.ACTION_PRESS:
            self._should_close = True

    def handle_events(self) -> bool:
        return not (self.window.is_closing or self._should_close)

    def start(self) -> None:
        shader = Utils.create_standard_shader(self.ctx)
        self.default_texture = Texture(self.ctx, fallback="white")
        shader.set_texture(self.default_texture, "maintex")
        self.sphere_mesh: Mesh = LoadObj.load_asmesh("assets/models/LowPolySphere.obj", shader)
        self.sphere_mesh.instance_attributes = [
            ('instance_model', 16),
            ('instance_color', 3),
        ]
        self.galaxy_particles: list[Galaxy.GalaxyParticle] = self.create_particles()
        self.instance_data = np.zeros((len(self.galaxy_particles), 19), dtype=np.float32)
        self.instance_data[:, 15] = 1.0

    def get_galaxy_angle(self, i) -> float:
        radius: float = (i / len(self.galaxy_particles)) * self.galaxy.radius
        arm: float = i % self.galaxy.arms
        angle: float = (radius / self.galaxy.radius) * 4.0 * math.pi
        angle += random.gauss(0, self.galaxy.distribution.x) + random.gauss(0, self.galaxy.distribution.y)
        return angle + arm * 2.0 * math.pi / self.galaxy.arms

    def create_particles(self, particle_count: int = 1000) -> list[Galaxy.GalaxyParticle]:
        particles: list[Galaxy.GalaxyParticle] = [
            Galaxy.GalaxyParticle(Vector3(0, 0, 0), Vector3(0, 0, 0), 0)
            for _ in range(particle_count)
        ]
        self.galaxy_particles = particles

        for i in range(particle_count):
            angle: float = self.get_galaxy_angle(i)
            radius: float = (i / len(self.galaxy_particles)) * self.galaxy.radius

            x: float = math.sin(angle) * radius
            y: float = random.gauss(
                0,
                (self.galaxy.distribution.x + self.galaxy.distribution.y) / 2.0
            ) * 0.01 * self.galaxy.radius
            z: float = math.cos(angle) * radius

            position: Vector3 = Vector3(x, y, z)

            particles[i] = Galaxy.GalaxyParticle(
                position,
                Vector3(0.0),
                0.005 * self.galaxy.radius,
                Vector3(1.0, 1.0, 1.0)
            )

        return particles

    def update_particles(self, delta_time: float, audio_features: dict[str, float | str]) -> None:
        angular_speed: float = self.galaxy.speed * delta_time

        for i, particle in enumerate(self.galaxy_particles):
            radius: float = (i / len(self.galaxy_particles)) * self.galaxy.radius
            angle: float = math.atan2(particle.position.x, particle.position.z)
            
            radial_factor: float = radius / self.galaxy.radius
            fixed_particle: bool = i % 2 < int(radial_factor * 2)
            if not fixed_particle:
                angle += angular_speed * (len(self.galaxy_particles) / (i + 1))
            else:
                angle += angular_speed

            particle.position.x = math.sin(angle) * radius
            particle.position.z = math.cos(angle) * radius

    def render_particles(
        self,
        view_matrix: list[list[float]],
        projection_matrix: list[list[float]]
    ) -> None:
        for i, particle in enumerate(self.galaxy_particles):
            model_matrix = Utils.create_model_matrix(
                particle.position,
                Rotation.identity(),
                Vector3(particle.scale)
            )

            self.instance_data[i, :16] = Utils.flatten_matrix(model_matrix)
            self.instance_data[i, 16:19] = (
                particle.colour.x,
                particle.colour.y,
                particle.colour.z,
            )

        self.sphere_mesh.update_instance_data(self.instance_data)

        self.sphere_mesh.draw_instanced(
            view_matrix,
            projection_matrix,
            len(self.galaxy_particles)
        )

    def update(self, audio_features: dict[str, float | str]) -> None:
        delta_time: float = self.delta_time

        self.update_particles(delta_time, audio_features)
        
        print(f"FPS: {1.0 / delta_time if delta_time > 0 else 0.0}")

    def render(self) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.window.clear(*self.clear_colour)  # type: ignore

        view_matrix: list[list[float]] = Utils.get_view_matrix(
            Vector3(0.0, 75.0, 100.0),
            Rotation.from_euler("xyz", [-50.0, 0.0, 0.0], degrees=True)
        )

        projection_matrix: list[list[float]] = Utils.get_projection_matrix(
            math.radians(90),
            self.width / self.height,
        )

        self.render_particles(view_matrix, projection_matrix)
        self.window.swap_buffers()

    def teardown(self) -> None:
        self.window.close()