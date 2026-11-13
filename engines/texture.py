from moderngl import Context, Texture as MGLTexture
import struct
from PIL import Image

class AssignedUnits:
    __instance: "AssignedUnits | None" = None
    def __init__(self) -> None:
        self.assigned = []

    def assign(self) -> int:
        unit = 0
        while unit in self.assigned:
            unit += 1
        self.assigned.append(unit)
        return unit
    
    def release(self, unit: int) -> None:
        if unit in self.assigned:
            self.assigned.remove(unit)

    def __new__(cls) -> "AssignedUnits":
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.assigned = []
        return cls.__instance
        

class Texture: 
    def __init__(self, ctx: Context, data: bytes | None = None, path: str|None = "", fallback: str = "white", size: tuple[int, int] = (1, 1)) -> None:
        self.__ctx: Context = ctx
        self.unit = AssignedUnits().assign()
        if data is None and not path:
            self.source = self.__generate_default_texture(fallback)
        elif path:
            self.source = self.from_path(path)
        elif data is not None:
            self.source = self.from_data(data, size)
        
    
    def __generate_default_texture(self, type: str = "white"
    ) -> MGLTexture:
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
            texture = self.__ctx.texture((2, 2), 4, data, alignment=1)
            return texture
        else:
            # Default to white texture if unknown type
            data: bytes = struct.pack("4B", 255, 255, 255, 255)

        return self.__ctx.texture((1, 1), 4, data, alignment=1)
        
    def from_path(self, file_path: str) -> MGLTexture:
        with Image.open(file_path) as image:
            rgba: Image.Image = image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            self.source = self.__ctx.texture(rgba.size, 4, rgba.tobytes(), alignment=1)
            return self.source

    def from_data(self, data: bytes, size: tuple[int, int] = (1, 1)) -> MGLTexture:
        self.source = self.__ctx.texture(size, 4, data, alignment=1)
        return self.source
    
    def __del__(self) -> None:
        AssignedUnits().release(self.unit)
        try:
            self.source.release()
        except Exception:
            pass