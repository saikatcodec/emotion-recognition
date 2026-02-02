import logging


logfile="logs/log-file.log"
kwargs = {
    "format": "%[ (asctime)s ] - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %I:%M:%S %p",
    "level": logging.INFO,
    "handlers": [
        logging.FileHandler(logfile),
    ],
}

logging.basicConfig(**kwargs)
