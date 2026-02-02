import logging
import os

logfile="logs/log-file.log"
os.makedirs(os.path.dirname(logfile), exist_ok=True)

kwargs = {
    "format": "[ %(asctime)s ] - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %I:%M:%S %p",
    "level": logging.INFO,
    # "filename": logfile
}

logging.basicConfig(**kwargs)
