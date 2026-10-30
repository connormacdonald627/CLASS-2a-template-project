import moderngl
import math
import numpy as np
import struct
from engines.Mesh import Mesh

class Utils:
    @staticmethod
    def create_debug_shader(ctx: moderngl.Context) -> moderngl.Program:
        vertex_shader_source: str = """
        #version 330
        in vec3 in_position;
        in vec3 in_color;
        uniform mat4 mvp;
        out vec3 vertex_color;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            vertex_color = in_color;
        }
        """
        fragment_shader_source: str = """
        #version 330
        out vec4 fragColor;
        in vec3 vertex_color;
        void main() {
            fragColor = vec4(vertex_color, 1.0);
        }
        """
        return ctx.program(vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source)

    @staticmethod
    def create_cube_mesh(
        shader_program: moderngl.Program
    ) -> Mesh:
        """Create a cube with one RGB color for each corner."""

        indexes: list[int] = [
            0, 3, 2, 0, 2, 1,
            4, 5, 6, 4, 6, 7,
            3, 7, 6, 3, 6, 2,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            0, 4, 7, 0, 7, 3,
        ]
        positions: list[tuple[float, float, float]] = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0)
        ]

        colors: list[tuple[float, float, float]] = [tuple((value + 1.0) / 2.0 for value in position) for position in positions]

        vertices = [(*position, *color) for position, color in zip(positions, colors)]
        return Mesh(vertices, indexes, shader_program, '3f 3f', ('in_color',))


    @staticmethod
    def get_projection_matrix(fov: float, aspect: float, near: float, far: float, orthographic: bool = False) -> list[list[float]]:
        if orthographic:
            right: float = aspect
            left: float = -aspect
            top: float = 1.0
            bottom: float = -1.0
            return [
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0]
            ]
        else:
            field_of_view: float = 1.0 / math.tan(fov / 2.0)
            return [
                [field_of_view / aspect, 0.0, 0.0, 0.0],
                [0.0, field_of_view, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0]
            ]

    @staticmethod
    def lerp(start: float, end: float, t: float) -> float:
        return start + (end - start) * t

    @staticmethod
    def create_model_view_matrix(translation: tuple[float, float, float], rotation: tuple[float, float, float], scale: tuple[float, float, float], view_projection: list[list[float]]):
        """Returns a model-view-projection matrix based on translation, rotation, scale, and view projection."""
        
        tx, ty, tz = translation
        rx, ry, rz = rotation
        sx, sy, sz = scale

        translation_matrix = np.array([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        cos_rx, sin_rx = math.cos(rx), math.sin(rx)
        cos_ry, sin_ry = math.cos(ry), math.sin(ry)
        cos_rz, sin_rz = math.cos(rz), math.sin(rz)
        rotation_matrix = np.array([
            [cos_ry * cos_rz, -cos_ry * sin_rz, sin_ry, 0.0],
            [cos_rx * sin_rz + sin_rx * sin_ry * cos_rz, cos_rx * cos_rz - sin_rx * sin_ry * sin_rz, -sin_rx * cos_ry, 0.0],
            [sin_rx * sin_rz - cos_rx * sin_ry * cos_rz, sin_rx * cos_rz + cos_rx * sin_ry * sin_rz, cos_rx * cos_ry, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        scale_matrix = np.array([
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        #Contruct MVP matrix based on translation, rotation, scale, and view projection
        model_matrix: np.ndarray = translation_matrix @ rotation_matrix @ scale_matrix
        view_projection_matrix: np.ndarray = np.array(view_projection, dtype=np.float32)
        model_view_projection_matrix: np.ndarray = view_projection_matrix @ model_matrix
        return model_view_projection_matrix.tolist()