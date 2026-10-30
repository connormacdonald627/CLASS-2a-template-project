import moderngl
import struct

class Mesh:
    def __init__(
        self,
        vertices: list[tuple[float, ...]] | None = None,
        indexes: list[int] | None = None,
        shader_program: moderngl.Program | None = None,
        vertex_format: str = '3f',
        attributes: tuple[str, ...] = (),
    ):
        self.__vbo: moderngl.Buffer | None = None
        self.__ibo: moderngl.Buffer | None = None
        self.__vao: moderngl.VertexArray | None = None
        self.__indexes: list[int] | None = indexes
        self.__vertices = vertices
        self.shader_program: moderngl.Program | None = shader_program
        self.__vertex_format = vertex_format
        self.__attributes = attributes
        self.position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __setup_buffers(self) -> None:
        if self.shader_program is None or not self.__vertices:
            return

        ctx = self.shader_program.ctx
        vertex_data = [value for vertex in self.__vertices for value in vertex]
        self.__vbo = ctx.buffer(struct.pack(f'{len(vertex_data)}f', *vertex_data))
        if self.__indexes is not None:
            self.__ibo = ctx.buffer(struct.pack(f'{len(self.__indexes)}I', *self.__indexes))

        content = [(self.__vbo, self.__vertex_format, 'in_position', *self.__attributes)]
        self.__vao = ctx.vertex_array(self.shader_program, content, index_buffer=self.__ibo)

    def draw(self, view_projection: list[list[float]]) -> None:
        if self.shader_program is None:
            return
        if self.__vao is None:
            self.__setup_buffers()
        if self.__vao is None:
            return

        if 'mvp' in self.shader_program:
            from engines.Utils import Utils

            mvp = Utils.create_model_view_matrix(self.position, self.rotation, self.scale, view_projection)
            matrix_data = (mvp[row][column] for column in range(4) for row in range(4))
            self.shader_program['mvp'].write(struct.pack('16f', *matrix_data))
        self.__vao.render(moderngl.TRIANGLES)