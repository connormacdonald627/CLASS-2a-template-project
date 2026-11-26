import moderngl
import struct
import time
import numpy as np
from scipy.spatial.transform import Rotation
from engines.shader import Shader
from engines.vector import Vector3

class Mesh:
    def __init__(
        self,
        vertex_data: list[tuple[float, ...]] | None = None,
        indicies: list[int] | None = None,
        shader: Shader | None = None,
        attributes: list[tuple[str, int]] = [
            ('in_position', 3),
            ('in_colour', 3),
            ('in_uv', 2),
            ('in_normal', 3),
        ],
        instance_data: list[tuple[float, ...]] | None = None,
        instance_attributes: list[tuple[str, int]] = []
    ) -> None:
        from engines.utils import Transform

        self.attributes: list[tuple[str, int]] = attributes.copy()
        self.instance_attributes: list[tuple[str, int]] = instance_attributes.copy()
        self.__vbo: moderngl.Buffer | None = None
        self.__ibo: moderngl.Buffer | None = None
        self.__vao: moderngl.VertexArray | None = None
        self.__instance_buffer: moderngl.Buffer | None = None
        self.__instance_data = instance_data
        self.__indicies: list[int] | None = indicies
        self.__vertex_data = vertex_data
        self.shader: Shader | None = shader
        self.transform: Transform = Transform()

    def __validate_attributes(self, attributes: list[tuple[str, int]] | None = None) -> None:
        if not attributes:
            raise ValueError("Attributes must be provided")
        for name, size in attributes:
            if not name:
                raise ValueError("Attribute name must be provided")
            if size not in (1, 2, 3, 4, 16):
                raise ValueError("Attribute size must be between 1 and 4, or 16")
            
        attribute_names = [name for name, _ in attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("Attribute names must be unique")


    def __format_attributes(self, attributes: list[tuple[str, int]]) -> tuple[str, tuple[str, ...]]:
        if self.shader is None:
            return '', ()
        program = self.shader.program
        # Skip attributes that are not in the shader program
        attribute_format = ' '.join(
            f'{size}f' if name in program else f'{size * 4}x'
            for name, size in attributes
        )
        names = tuple(name for name, _ in attributes if name in program)
        return attribute_format, names

    def __setup_buffers(self) -> None:
        if self.shader is None or not self.__vertex_data:
            return
        self.__setup_vertex_attributes()
        self.__setup_instance_attributes()
        if self.__vbo is None:
            return
        vertex_format, names = self.__format_attributes(self.attributes)
        content = [(self.__vbo, vertex_format, *names)]
        if self.__instance_buffer is not None:
            instance_format, instance_names = self.__format_attributes(self.instance_attributes)
            if instance_names:
                content.append((self.__instance_buffer, f'{instance_format} /i', *instance_names))
        self.__vao = self.shader.ctx.vertex_array(self.shader.program, content, index_buffer=self.__ibo)
        
    def __setup_vertex_attributes(self) -> None:
        if self.shader is None or not self.__vertex_data:
            return
        self.__validate_attributes(self.attributes)
            
        ctx = self.shader.ctx
        vertex_data = [value for vertex in self.__vertex_data for value in vertex]
        data = struct.pack(f'{len(vertex_data)}f', *vertex_data)
        if self.__vbo is None:
            self.__vbo = ctx.buffer(data)
        else:
            self.__vbo.write(data)
        if self.__indicies is not None and self.__ibo is None:
            self.__ibo = ctx.buffer(struct.pack(f'{len(self.__indicies)}I', *self.__indicies))
            
    def __setup_instance_attributes(self) -> None:
        if self.shader is None or self.__instance_data is None or len(self.__instance_data) == 0:
            return
        self.__validate_attributes(self.instance_attributes)
        ctx = self.shader.ctx
        instance_data = [value for instance in self.__instance_data for value in instance]
        data = struct.pack(f'{len(instance_data)}f', *instance_data)
        if self.__instance_buffer is None:
            self.__instance_buffer = ctx.buffer(data)
            if self.__vao is not None:
                self.__vao.release()
                self.__vao = None
        else:
            self.__instance_buffer.write(data)
        
    def update_vertex_data(self, vertex_data: np.ndarray) -> None:
        self.__vertex_data = vertex_data
        self.__setup_vertex_attributes()
        
    def update_instance_data(self, instance_data: np.ndarray) -> None:
        self.__instance_data = instance_data
        self.__setup_instance_attributes()

    def draw(self, view: list[list[float]], projection: list[list[float]]) -> None:
        if self.shader is None:
            return
        if self.__vao is None:
            self.__setup_buffers()
        if self.__vao is None or self.__vbo is None:
            return

        self.__set_matrices(view, projection)
        self.__vao.render(moderngl.TRIANGLES)
        
    def draw_instanced(self, view: list[list[float]], projection: list[list[float]], instance_count: int) -> None:
        if instance_count < 0:
            raise ValueError("Instance count cannot be negative")
        if self.__instance_data is not None and instance_count > len(self.__instance_data):
            raise ValueError("Instance count exceeds the available instance data")
        if instance_count == 0:
            return
        if self.shader is None:
            return
        if self.__vao is None:
            self.__setup_buffers()
        if self.__vao is None or self.__vbo is None:
            return

        self.__set_matrices(view, projection)
        self.__vao.render(moderngl.TRIANGLES, instances=instance_count)
            
    
    def __set_matrices(self, view: list[list[float]], projection: list[list[float]]) -> None:
        if self.shader is None:
            return
        from engines.utils import Utils
        
        # Flatten the MVP matrix for writing to the shader uniform to a 16 length 1D array
        # And transpose the matrix to match the column-major order expected by OpenGL

        self.shader.program['model'] = Utils.flatten_matrix(Utils.create_model_matrix(self.transform.position, self.transform.rotation, self.transform.scale))
        self.shader.program['view'] = Utils.flatten_matrix(view)
        self.shader.program['projection'] = Utils.flatten_matrix(projection)
            
        
    def __del__(self) -> None:
        try:
            if self.__vao is not None:
                self.__vao.release()
                self.__vao = None
            if self.__vbo is not None:
                self.__vbo.release()
                self.__vbo = None
            if self.__instance_buffer is not None:
                self.__instance_buffer.release()
                self.__instance_buffer = None
            if self.__ibo is not None:
                self.__ibo.release()
                self.__ibo = None
        except Exception:
            pass