import logging

# Add configuration for both
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[
        logging.FileHandler("log-file.log"),
        logging.StreamHandler(),
    ],
)


print(logging.getLogger(__name__))
