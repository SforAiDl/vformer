def pair(t):
    """
    Parameters
    ----------
    t: tuple[int] or int
    """
    return t if isinstance(t, tuple) else (t, t)
