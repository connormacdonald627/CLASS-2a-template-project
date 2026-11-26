import moderngl
import math
import numpy as np
from scipy.spatial.transform import Rotation
from engines.shader import Shader
from engines.mesh import Mesh
from engines.vector import Vector3, Vector2

class Transform:
    # Transform class to handle position, rotation, and scale of objects
    def __init__(self) -> None:
        self.position: Vector3 = Vector3(0.0, 0.0, 0.0)
        self.rotation: Rotation = Rotation.identity()
        self.scale: Vector3 = Vector3(1.0, 1.0, 1.0)
        
    def get_matrix(self) -> list[list[float]]:
        from engines.utils import Utils
        return Utils.create_model_matrix(self.position, self.rotation, self.scale)

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
    def get_view_matrix(position: Vector3, rotation: Rotation) -> list[list[float]]:
        """Create a view matrix based on position and rotation"""
        x, y, z = position
        rot_matrix = rotation.as_matrix().astype(np.float32)
        inverted_trans = Vector3(
            -(rot_matrix[0, 0] * x + rot_matrix[1, 0] * y + rot_matrix[2, 0] * z),
            -(rot_matrix[0, 1] * x + rot_matrix[1, 1] * y + rot_matrix[2, 1] * z),
            -(rot_matrix[0, 2] * x + rot_matrix[1, 2] * y + rot_matrix[2, 2] * z),
        )

        view_matrix = np.array(
            [
                [rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2], 0.0],
                [rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2], 0.0],
                [rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2], 0.0],
                [inverted_trans.x, inverted_trans.y, inverted_trans.z, 1.0],
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
    def create_model_matrix(
        position: Vector3,
        rotation: Rotation,
        scale: Vector3
    ) -> list[list[float]]:
        """Create a model matrix based on position, rotation, and scale"""

        x, y, z = position
        rot_matrix = rotation.as_matrix().astype(np.float32)
        scale_x, scale_y, scale_z = scale

        model_matrix = np.array(
            [
                [rot_matrix[0, 0] * scale_x, rot_matrix[1, 0] * scale_x, rot_matrix[2, 0] * scale_x, 0.0],
                [rot_matrix[0, 1] * scale_y, rot_matrix[1, 1] * scale_y, rot_matrix[2, 1] * scale_y, 0.0],
                [rot_matrix[0, 2] * scale_z, rot_matrix[1, 2] * scale_z, rot_matrix[2, 2] * scale_z, 0.0],
                [x, y, z, 1.0],
            ],
            dtype=np.float32,
        )
        return model_matrix.tolist()

    @staticmethod
    def flatten_matrix(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
        """Flatten a 4x4 matrix to a 1D list in column-major order"""
        return np.asarray(matrix, dtype=np.float32).T.flatten()