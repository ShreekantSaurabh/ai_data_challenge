import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    :param name: Namespace used to identify the logger.
    :returns: Logger configured with a stream handler and default formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        f = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        h.setFormatter(f)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
