import moderngl
import struct
from scipy.spatial.transform import Rotation

class Mesh:
    def __init__(
        self,
        vertex_data: list[tuple[float, ...]] | None = None,
        indexes: list[int] | None = None,
        shader_program: moderngl.Program | None = None,
        attributes: list[tuple[str, int]] = [
            ('in_position', 3),
            ('in_colour', 3),
            ('in_uv', 2),
            ('in_normal', 3),
        ],
    ) -> None:
        self.attributes: list[tuple[str, int]] = attributes.copy()
        self.__vbo: moderngl.Buffer | None = None
        self.__ibo: moderngl.Buffer | None = None
        self.__vao: moderngl.VertexArray | None = None
        self.__indexes: list[int] | None = indexes
        self.__vertex_data = vertex_data
        self.shader_program: moderngl.Program | None = shader_program
        self.position: list[float] = [0.0, 0.0, 0.0]
        self.rotation: Rotation = Rotation.identity()
        self.scale: list[float] = [1.0, 1.0, 1.0]
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
        if self.shader_program is None or not self.__vertex_data:
            return
        self.__validate_attributes()

        ctx = self.shader_program.ctx
        vertex_data = [value for vertex in self.__vertex_data for value in vertex]
        self.__vbo = ctx.buffer(struct.pack(f'{len(vertex_data)}f', *vertex_data))
        if self.__indexes is not None:
            self.__ibo = ctx.buffer(struct.pack(f'{len(self.__indexes)}I', *self.__indexes))

        vertex_format = ' '.join(
            f'{size * 4}x' if name == 'in_normal' and name not in self.shader_program else f'{size}f'
            for name, size in self.attributes
        )
        names = tuple(
            name for name, _ in self.attributes
            if name in self.shader_program
        )
        content = [(self.__vbo, vertex_format, *names)]
        self.__vao = ctx.vertex_array(self.shader_program, content, index_buffer=self.__ibo)

    def draw(self, view_projection: list[list[float]]) -> None:
        if self.shader_program is None:
            return
        if self.__vao is None:
            self.__setup_buffers()
        if self.__vao is None:
            return

        if 'mvp' in self.shader_program:
            from engines.utils import Utils

            mvp = Utils.create_model_view_matrix(self.position, self.rotation, self.scale, view_projection)
            matrix_data = (mvp[row][column] for column in range(4) for row in range(4))
            uniform = self.shader_program['mvp']
            if isinstance(uniform, moderngl.Uniform):
                uniform.write(struct.pack('16f', *matrix_data))
        self.__vao.render(moderngl.TRIANGLES)