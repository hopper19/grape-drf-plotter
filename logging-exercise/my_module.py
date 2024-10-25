import logging 
logger = logging.getLogger(__name__)


def my_function():
    logger.info("This is a log message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.critical("This is a critical error")