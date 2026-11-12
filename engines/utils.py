import moderngl
import math
import numpy as np
import struct
from PIL import Image
from scipy.spatial.transform import Rotation
from engines.mesh import Mesh


class Utils:
    @staticmethod
    def load_file(file_path: str, binary: bool = False) -> str | bytes:
        mode: str = "rb" if binary else "r"
        with open(file_path, mode) as f:
            return f.read()

    @staticmethod
    def create_standard_shader(ctx: moderngl.Context) -> moderngl.Program:
        vertex_shader_source: str = str(
            Utils.load_file("assets/shaders/standard_vert.glsl")
        )
        fragment_shader_source: str = str(
            Utils.load_file("assets/shaders/standard_frag.glsl")
        )
        return ctx.program(
            vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
        )

    @staticmethod
    def load_texture(ctx: moderngl.Context, file_path: str) -> moderngl.Texture:
        with Image.open(file_path) as image:
            rgba: Image.Image = image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            return ctx.texture(rgba.size, 4, rgba.tobytes(), alignment=1)

    @staticmethod
    def generate_default_texture(
        ctx: moderngl.Context, type: str = "white"
    ) -> moderngl.Texture:
        if type == "white":
            data: bytes = struct.pack("4B", 255, 255, 255, 255)
        elif type == "black":
            data: bytes = struct.pack("4B", 0, 0, 0, 255)
        elif type == "normal":
            data: bytes = struct.pack("4B", 128, 128, 255, 255)
        elif type == "checkerboard":
            purple: bytes = struct.pack("4B", 255, 0, 255, 255)
            black: bytes = struct.pack("4B", 0, 0, 0, 255)
            data: bytes = purple + black + black + purple
            texture = ctx.texture((2, 2), 4, data, alignment=1)
            return texture
        else:
            # Default to white texture if unknown type
            data: bytes = struct.pack("4B", 255, 255, 255, 255)

        return ctx.texture((1, 1), 4, data, alignment=1)

    @staticmethod
    def set_shader_texture(
        shader: moderngl.Program,
        texture: moderngl.Texture,
        sampler_name: str,
        texture_unit: int = 0,
        sampling_filter: tuple[int, int] = (moderngl.NEAREST, moderngl.NEAREST),
        wrap_filter: tuple[bool, bool] = (True, True),
    ) -> None:
        """Bind a texture to a texture sampler"""
        try:
            texture.use(location=texture_unit)
            texture.filter = sampling_filter
            texture.repeat_x = wrap_filter[0]
            texture.repeat_y = wrap_filter[1]
            shader[sampler_name] = texture_unit
        except:
            pass

    @staticmethod
    def create_cube_mesh(shader_program: moderngl.Program) -> Mesh:
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
        return Mesh(vertex_data, indexes, shader_program)

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
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0],
            ]
        else:
            field_of_view: float = 1.0 / math.tan(fov / 2.0)
            return [
                [field_of_view / aspect, 0.0, 0.0, 0.0],
                [0.0, field_of_view, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    (far + near) / (near - far),
                    (2 * far * near) / (near - far),
                ],
                [0.0, 0.0, -1.0, 0.0],
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
                [rot_matrix[0, 0] * scale_x, rot_matrix[0, 1] * scale_y, rot_matrix[0, 2] * scale_z, x],
                [rot_matrix[1, 0] * scale_x, rot_matrix[1, 1] * scale_y, rot_matrix[1, 2] * scale_z, y],
                [rot_matrix[2, 0] * scale_x, rot_matrix[2, 1] * scale_y, rot_matrix[2, 2] * scale_z, z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Construct the MVP matrix from the model and view-projection matrices.
        return (np.asarray(view_projection, dtype=np.float32) @ model_matrix).tolist()
