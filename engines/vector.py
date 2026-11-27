from typing import Any
import numpy as np
    
# Vector class so we dont have to manually handle numpy vectors
class Vector:

    # Allow iteration over vector values
    def __iter__(self) -> Any:
        return iter(self.__dict__.values())

    def length(self) -> float:
        return float(np.linalg.norm(self.__as_array()))

    def dot(self, other) -> float:
        return float(np.dot(self.__as_array(), self.__get_array(other)))

    def cross(self, other):
        return self.__class__(*(
            np.cross(self.__as_array(), self.__get_array(other))
        ))
    
    def lerp(self, other, t):
        return self.__class__(*(
            (1 - self.__get_array(t)) * self.__as_array() + self.__get_array(t) * self.__get_array(other)
        ))

    def __as_array(self) -> np.ndarray:
        return np.asarray(tuple(self))

    def __get_array(self, other) -> Any:
        if isinstance(other, Vector):
            return other.__as_array()
        return other

    def __gt__(self, other) -> bool:
        return bool(np.all(self.__as_array() > self.__get_array(other)))

    def __ge__(self, other) -> bool:
        return bool(np.all(self.__as_array() >= self.__get_array(other)))

    def __le__(self, other) -> bool:
        return bool(np.all(self.__as_array() <= self.__get_array(other)))

    def __lt__(self, other) -> bool:
        return bool(np.all(self.__as_array() < self.__get_array(other)))
    
    def __eq__(self, other) -> bool:
        return bool(np.allclose(self.__as_array(), self.__get_array(other)))

    def __add__(self, other):
        return self.__class__(*(
            self.__as_array() + self.__get_array(other)
        ))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__class__(*(
            self.__as_array() - self.__get_array(other)
        ))

    def __rsub__(self, other):
        return self.__class__(*(
            self.__get_array(other) - self.__as_array()
        ))

    def __mul__(self, other):
        return self.__class__(*(
            self.__as_array() * self.__get_array(other)
        ))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__class__(*(
            self.__as_array() / self.__get_array(other)
        ))

    def __rtruediv__(self, other):
        return self.__class__(*(
            self.__get_array(other) / self.__as_array()
        ))

    def __neg__(self):
        return self.__class__(*(
            -self.__as_array()
        ))


class Vector4(Vector):
    def __init__(
        self, x: float, y: float | None = None, z: float | None = None, w: float | None = None
    ) -> None:
        if y is None and z is None and w is None:
            self.x = x
            self.y = x
            self.z = x
            self.w = x
        else:
            self.x = x
            self.y = y if y is not None else 0.0
            self.z = z if z is not None else 0.0
            self.w = w if w is not None else 0.0


class Vector3(Vector):
    def __init__(
        self, x: float, y: float | None = None, z: float | None = None
    ) -> None:
        if y is None and z is None:
            self.x = x
            self.y = x
            self.z = x
        else:
            self.x = x
            self.y = y if y is not None else 0.0
            self.z = z if z is not None else 0.0


class Vector2(Vector):
    def __init__(
        self, x: float, y: float | None = None
    ) -> None:
        if y is None:
            self.x = x
            self.y = x
        else:
            self.x = x
            self.y = y