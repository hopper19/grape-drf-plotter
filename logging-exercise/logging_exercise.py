# Learning how to use logging in Python
# Author: Cuong Nguyen

import os
import logging
import my_module
from datetime import datetime
logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        filename="exercise.log",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-10s | %(name)-15s | %(funcName)-30s | %(message)s",
    )
    # logger.info("This is a log message")
    # logger.warning("This is a warning")
    # my_module.my_function()
    # logger.error("This is an error")
    # logger.critical("This is a critical error")


if __name__ == "__main__":
    main()
