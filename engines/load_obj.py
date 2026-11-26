from engines.vector import Vector2, Vector3
from engines.utils import Utils
from engines.mesh import Mesh
from engines.shader import Shader

class LoadObj:
    # Loads an OBJ file and returns its vertices, indices, UVs, and normals as a dictionary
    # v = vertex
    # vt = texture coordinate/uv
    # vn = normal
    # vc = colour
    # f = face (vertex/uv/normal)

    @staticmethod
    def load_asmesh(file_path: str, shader: Shader) -> Mesh:
        data: dict[str, list] = LoadObj.load(file_path)
        
        vertex_data = [
            (
                *data['verticies'][i],
                *data['colours'][i],
                *data['uvs'][i],
                *data['normals'][i],
            )
            for i in range(len(data['verticies']))
        ]
        return Mesh(vertex_data=vertex_data, indicies=data['indices'], shader=shader)

    @staticmethod
    def load(file_path: str) -> dict[str, list[Vector3] | list[Vector2] | list[int]]:
        if ".obj" not in file_path:
            raise ValueError("File must be an OBJ file")
        
        verticies: list[Vector3] = []
        indices: list[int] = []
        colours: list[Vector3] = []
        uvs: list[Vector2] = []
        new_uvs: list[Vector2] = []
        normals: list[Vector3] = []
        new_normals: list[Vector3] = []

        for lines in str(Utils.load_file(file_path)).splitlines():
            if lines.startswith('v '):
                _, x, y, z = lines.split()
                verticies.append(Vector3(float(x), float(y), float(z)))
            elif lines.startswith('vt '):
                _, u, v = lines.split()
                uvs.append(Vector2(float(u), float(v)))
            elif lines.startswith('vn '):
                _, x, y, z = lines.split()
                normals.append(Vector3(float(x), float(y), float(z)))
            elif lines.startswith('vc '):
                _, r, g, b = lines.split()
                colours.append(Vector3(float(r), float(g), float(b)))
            elif lines.startswith('f '):
                if len(new_uvs) == 0 or len(new_normals) == 0:
                    new_uvs = [Vector2(0, 0) for _ in range(len(verticies))]
                    new_normals = [Vector3(0, 0, 0) for _ in range(len(verticies))]
                    
                _, *face = lines.split()
                faces: list[int] = []
                for part in face:
                    vertex, uv, normal = part.split('/')[:3]
                    vertex_index: int = int(vertex) - 1
                    
                    # remap uvs to be in correct order for verticies, same for normals too
                    if uv:
                        uv_index: int = int(uv) - 1
                        if 0 <= uv_index < len(uvs):
                            new_uvs[vertex_index] = uvs[uv_index]
                    if normal:
                        normal_index: int = int(normal) - 1
                        if 0 <= normal_index < len(normals):
                            new_normals[vertex_index] = normals[normal_index]
                        

                    if vertex_index >= 0:
                        index = vertex_index
                    elif vertex_index < -1:
                        index = len(verticies) + vertex_index + 1
                    else:
                        raise ValueError("OBJ indices cannot be zero")
                    faces.append(index)

                # Triangulate the face as a fan from its first vertex
                for i in range(1, len(faces) - 1):
                    indices.extend([faces[0], faces[i], faces[i + 1]])
        
        if new_uvs:
            uvs = new_uvs
        if new_normals:
            normals = new_normals
                    
        # Populate empty data if no verticies, colours, uvs, or normals were found
        if not verticies:
            verticies.extend(Vector3(0.0, 0.0, 0.0) for _ in range(len(verticies)))
        if not colours:
            colours.extend(Vector3(1.0, 1.0, 1.0) for _ in range(len(verticies)))
        if not uvs:
            uvs.extend(Vector2(0.0, 0.0) for _ in range(len(verticies)))
        if not normals:
            normals.extend(Vector3(0.0, 0.0, 0.0) for _ in range(len(verticies)))


        return {
            'verticies': verticies,
            'colours': colours,
            'indices': indices,
            'uvs': uvs,
            'normals': normals
        }