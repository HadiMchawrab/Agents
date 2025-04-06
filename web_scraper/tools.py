import logging
import time
from contextlib import contextmanager


def get_logger(name: str = "web_scraper"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def timer(name="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMER] {name} took {end - start:.2f}s")