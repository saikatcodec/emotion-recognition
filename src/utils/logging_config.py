import logging

__all__ = ["configure_logging"]


def configure_logging(level=None, logfile="output/log-file.log"):
    root = logging.getLogger()
    if root.handlers:
        return

    # Set default level to INFO if not provided
    if level is None:
        level = logging.INFO

    kwargs = {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %I:%M:%S %p",
        "level": level,
        "handlers": [
            logging.FileHandler(logfile),
        ],
    }

    logging.basicConfig(**kwargs)
