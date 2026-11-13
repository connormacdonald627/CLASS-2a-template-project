import moderngl
import struct
from engines.shader import Shader
from scipy.spatial.transform import Rotation
from engines.vector import Vector3
import numpy as np

class Mesh:
    def __init__(
        self,
        vertex_data: list[tuple[float, ...]] | None = None,
        indexes: list[int] | None = None,
        shader: Shader | None = None,
        attributes: list[tuple[str, int]] = [
            ('in_position', 3),
            ('in_colour', 3),
            ('in_uv', 2),
            ('in_normal', 3),
        ],
    ) -> None:
        from engines.utils import Transform

        self.attributes: list[tuple[str, int]] = attributes.copy()
        self.__vbo: moderngl.Buffer | None = None
        self.__ibo: moderngl.Buffer | None = None
        self.__vao: moderngl.VertexArray | None = None
        self.__indexes: list[int] | None = indexes
        self.__vertex_data = vertex_data
        self.shader: Shader | None = shader
        self.transform: Transform = Transform()
        self.__validate_attributes()

    def __validate_attributes(self) -> None:
        if not self.attributes:
            raise ValueError("Attributes must be provided")
        for name, size in self.attributes:
            if not name:
                raise ValueError("Attribute name must be provided")
            if size < 1 or size > 4:
                raise ValueError("Attribute size must be between 1 and 4")
            
        attribute_names = [name for name, _ in self.attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("Attribute names must be unique")

    def __setup_buffers(self) -> None:
        if self.shader is None or not self.__vertex_data:
            return
        self.__validate_attributes()

        ctx = self.shader.ctx
        vertex_data = [value for vertex in self.__vertex_data for value in vertex]
        self.__vbo = ctx.buffer(struct.pack(f'{len(vertex_data)}f', *vertex_data))
        if self.__indexes is not None:
            self.__ibo = ctx.buffer(struct.pack(f'{len(self.__indexes)}I', *self.__indexes))

        # Skip attributes that are not in the shader program
        vertex_format = ' '.join(
            f'{size * 4}x' if name not in self.shader.program else f'{size}f'
            for name, size in self.attributes
        )
        names = tuple(
            name for name, _ in self.attributes
            if name in self.shader.program
        )
        content = [(self.__vbo, vertex_format, *names)]
        self.__vao = ctx.vertex_array(self.shader.program, content, index_buffer=self.__ibo)

    def draw(self, view_projection: list[list[float]]) -> None:
        if self.shader is None:
            return
        if self.__vao is None:
            self.__setup_buffers()
        if self.__vao is None:
            return

        # Set the model-view-projection matrix if the shader has an 'mvp' uniform
        if 'mvp' in self.shader.program:
            mvp = np.array(self.transform.get_matrix()) @ np.array(view_projection)
            matrix_data = (mvp[row][column] for column in range(4) for row in range(4))
            uniform = self.shader.program['mvp']
            if isinstance(uniform, moderngl.Uniform):
                uniform.write(struct.pack('16f', *matrix_data))
        self.__vao.render(moderngl.TRIANGLES)
        
    def __del__(self) -> None:
        try:
            if self.__vao is not None:
                self.__vao.release()
                self.__vao = None
            if self.__vbo is not None:
                self.__vbo.release()
                self.__vbo = None
            if self.__ibo is not None:
                self.__ibo.release()
                self.__ibo = None
        except Exception:
            pass