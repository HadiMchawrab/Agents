import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
import os

def get_logger(name: str = "web_scraper", log_file: str = "logs/web_scraper.log"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler.setFormatter(stream_formatter)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def timer(name="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"[TIMER] {name} took {end - start:.2f}s")
