import numpy as np

# Vector class so we dont have to manually handle numpy vectors
class Vector:
    # Allow iteration over our vector values from public params
    def __iter__(self):
        return iter(self.__dict__.values())

    def __get_array(self, other):
        if isinstance(other, Vector):
            return np.array(other)
        return other

    def __add__(self, other):
        return self.__class__(*(np.array(self) + self.__get_array(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__class__(*(np.array(self) - self.__get_array(other)))

    def __rsub__(self, other):
        return self.__class__(*(self.__get_array(other) - np.array(self)))

    def __mul__(self, other):
        return self.__class__(*(np.array(self) * self.__get_array(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__class__(*(np.array(self) / self.__get_array(other)))

    def __rtruediv__(self, other):
        return self.__class__(*(self.__get_array(other) / np.array(self)))

    def __neg__(self):
        return self.__class__(*(-np.array(self)))


class Vector4(Vector):
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Vector3(Vector):
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

class Vector2(Vector):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y