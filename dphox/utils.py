def fix_dataclass_init_docs(cls):
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    Args:
        cls: The class whose docstring needs fixing

    Returns:
        The class that was passed so this function can be used as a decorator

    See Also:
        https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f'{cls.__name__}.__init__'
    return cls
