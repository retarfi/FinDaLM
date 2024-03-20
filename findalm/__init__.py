import logging


def get_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
