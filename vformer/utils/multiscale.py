import logging

logger = logging.get_logger(__name__)

def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)
  
