"""logging handler"""
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


class Logger:
    """Logger class"""

    @staticmethod
    def info(message):
        """info"""
        logging.info(message)

    @staticmethod
    def warning(message):
        """warning"""
        logging.warning(message)

    @staticmethod
    def debug(message):
        """debug"""
        logging.debug(message)

    @staticmethod
    def error(message):
        """error"""
        logging.error(message)

        