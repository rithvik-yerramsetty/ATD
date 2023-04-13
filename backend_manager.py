
import warnings
from typing import TypeAlias, Union, NewType

try:
    import numpy as Backend
    Matrix = NewType('Matrix', Backend.ndarray)
    Decimal = Union[int, float, Backend.floating, Backend.integer]
    Backend.create_matrix_func = Backend.array
    Backend.convert_to_matrix_func = Backend.asarray


    def extend_with_000(mat: Matrix) -> Matrix:
        """
        Pad 0 around the matrix.
        """
        return Backend.pad(mat, ((0, 1), (0, 1)))


    def extend_with_010(mat: Matrix) -> Matrix:
        """
        Setting the bottom right number to 1.
        """
        mat_ = extend_with_000(mat)
        mat_[-1, -1] = 1
        return mat_


    if Backend.__version__ < "1.19.0":
        warnings.warn(f"NumPy {Backend.__version__} might not work. You'd better upgrade to a newer version.",
                      category=ImportWarning)
except Exception:
    raise ImportError("Unable to import NumPy!")
else:
    print("Successfully initialized NumPy.\nUsing NumPy as Backend.")
