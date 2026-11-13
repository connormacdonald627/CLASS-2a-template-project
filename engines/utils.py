import moderngl
import math
import numpy as np
import struct
from scipy.spatial.transform import Rotation
from engines.mesh import Mesh
from engines.shader import Shader


class Utils:
    @staticmethod
    def load_file(file_path: str, binary: bool = False) -> str | bytes:
        mode: str = "rb" if binary else "r"
        with open(file_path, mode) as f:
            return f.read()

    @staticmethod
    def create_standard_shader(ctx: moderngl.Context) -> Shader:
        vertex_shader_source: str = str(
            Utils.load_file("assets/shaders/standard_vert.glsl")
        )
        fragment_shader_source: str = str(
            Utils.load_file("assets/shaders/standard_frag.glsl")
        )
        return Shader(ctx,
            vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
        )

    @staticmethod
    def create_cube_mesh(shader: Shader) -> Mesh:
        """Creates a cube mesh"""

        vertices: list[tuple[float, float, float]] = [
            (-1, -1, -1),
            (-1, 1, -1),
            (1, 1, -1),
            (1, -1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, 1, 1),
            (1, 1, -1),
            (-1, -1, -1),
            (1, -1, -1),
            (1, -1, 1),
            (-1, -1, 1),
            (1, -1, -1),
            (1, 1, -1),
            (1, 1, 1),
            (1, -1, 1),
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, 1),
            (-1, 1, -1),
        ]

        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] * 6
        normals = (
            [(0.0, 0.0, -1.0)] * 4
            + [(0.0, 0.0, 1.0)] * 4
            + [(0.0, 1.0, 0.0)] * 4
            + [(0.0, -1.0, 0.0)] * 4
            + [(1.0, 0.0, 0.0)] * 4
            + [(-1.0, 0.0, 0.0)] * 4
        )

        indexes: list[int] = [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15,
            16, 17, 18, 16, 18, 19,
            20, 21, 22, 20, 22, 23,
        ]

        vertex_data = [
            (*vertices[i], 1.0, 1.0, 1.0, *uvs[i], *normals[i])
            for i in range(len(vertices))
        ]
        return Mesh(vertex_data, indexes, shader)

    @staticmethod
    def get_view_matrix(position: list[float], rotation: Rotation) -> list[list[float]]:
        """Create a view matrix based on position and rotation"""
        x, y, z = position
        rot_matrix = rotation.as_matrix().astype(np.float32)
        inverted_trans = np.array(
            [
                -(rot_matrix[0, 0] * x + rot_matrix[1, 0] * y + rot_matrix[2, 0] * z),
                -(rot_matrix[0, 1] * x + rot_matrix[1, 1] * y + rot_matrix[2, 1] * z),
                -(rot_matrix[0, 2] * x + rot_matrix[1, 2] * y + rot_matrix[2, 2] * z),
            ],
            dtype=np.float32,
        )

        view_matrix = np.array(
            [
                [rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2], 0.0],
                [rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2], 0.0],
                [rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2], 0.0],
                [inverted_trans[0], inverted_trans[1], inverted_trans[2], 1.0],
            ]
        )
        return view_matrix.tolist()

    @staticmethod
    def get_projection_matrix(
        fov: float,
        aspect: float,
        near: float = 0.01,
        far: float = 1000.0,
        orthographic: bool = False,
    ) -> list[list[float]]:
        if orthographic:
            right: float = aspect
            left: float = -aspect
            top: float = 1.0
            bottom: float = -1.0
            return [
                [2.0 / (right - left), 0.0, 0.0, 0.0],
                [0.0, 2.0 / (top - bottom), 0.0, 0.0],
                [0.0, 0.0, -2.0 / (far - near), 0.0],
                [
                    -(right + left) / (right - left),
                    -(top + bottom) / (top - bottom),
                    -(far + near) / (far - near),
                    1.0,
                ],
            ]
        else:
            field_of_view: float = 1.0 / math.tan(fov / 2.0)
            return [
                [field_of_view / aspect, 0.0, 0.0, 0.0],
                [0.0, field_of_view, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), -1.0],
                [0.0, 0.0, (2 * far * near) / (near - far), 0.0],
            ]

    @staticmethod
    def lerp(start: float, end: float, t: float) -> float:
        return start + (end - start) * t

    @staticmethod
    def create_mvp(
        position: list[float],
        rotation: Rotation,
        scale: list[float],
        view_projection: list[list[float]],
    ) -> list[list[float]]:
        """Create a model-view-projection matrix based on position, rotation, scale, and view projection"""

        x, y, z = position
        rot_matrix = rotation.as_matrix().astype(np.float32)
        scale_x, scale_y, scale_z = np.asarray(scale, dtype=np.float32)

        model_matrix = np.array(
            [
                [rot_matrix[0, 0] * scale_x, rot_matrix[1, 0] * scale_x, rot_matrix[2, 0] * scale_x, 0.0],
                [rot_matrix[0, 1] * scale_y, rot_matrix[1, 1] * scale_y, rot_matrix[2, 1] * scale_y, 0.0],
                [rot_matrix[0, 2] * scale_z, rot_matrix[1, 2] * scale_z, rot_matrix[2, 2] * scale_z, 0.0],
                [x, y, z, 1.0],
            ],
            dtype=np.float32,
        )

        return (model_matrix @ np.asarray(view_projection, dtype=np.float32)).tolist()
