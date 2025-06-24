from typing import Any, get_args, get_origin


def get_generic_type(cls: type[Any], origin: type[Any], index: int = 0) -> type[Any]:
    """
    Returns the ``index``-th generic type of the superclass ``origin``.

    Args:
        cls: The class on which to inspect the superclasses.
        origin: The superclass to look for.
        index: The index of the generic type of the superclass.

    Returns
    -------
        The generic type.
    """
    for base in cls.__orig_bases__:  # type: ignore
        if get_origin(base) == origin:
            args = get_args(base)
            if not args:
                raise ValueError(f"`{cls.__name__}` does not provide a generic parameter for `{origin.__name__}`")
            return get_args(base)[index]
    raise ValueError(f"`{cls.__name__}` does not inherit from `{origin.__name__}`")
