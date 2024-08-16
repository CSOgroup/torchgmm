import sys
from os import PathLike
from typing import Union

if sys.version_info < (3, 9, 0):
    # PathLike is not generic for Python 3.9
    PathType = Union[str, PathLike]
else:
    PathType = Union[str, PathLike[str]]  # type: ignore
