from moderngl import Context
from engines.texture import Texture

class Shader:
    def __init__(
        self,
        ctx: Context,
        vertex_shader: str | bytes | None = None,
        fragment_shader: str | bytes | None = None,
        geometry_shader: str | bytes | None = None,
        tess_control_shader: str | bytes | None = None,
        tess_evaluation_shader: str | bytes | None = None,
        task_shader: str | bytes | None = None,
        mesh_shader: str | bytes | None = None,
    ) -> None:
        self.ctx = ctx
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader,
            task_shader=task_shader,
            mesh_shader=mesh_shader,
        )
        
    
    def set_texture(
        self,
        texture: Texture,
        sampler_name: str,
        texture_unit: int = 0,
        sampling_filter: tuple[int, int] = (Context.NEAREST, Context.NEAREST),
        wrap_filter: tuple[bool, bool] = (True, True),
    ) -> None:
        """Bind a texture to a texture sampler"""
        try:
            texture.source.use(location=texture_unit)
            texture.source.filter = sampling_filter
            texture.source.repeat_x = wrap_filter[0]
            texture.source.repeat_y = wrap_filter[1]
            self.program[sampler_name] = texture_unit
        except:
            pass
        
    def __del__(self) -> None:
        try:
            self.program.release()
        except Exception:
            pass
    
    