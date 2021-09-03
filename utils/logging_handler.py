import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

class Logger(object):
    @staticmethod
    def info(message):
        logging.info(message)
    
    @staticmethod
    def warning(message):
        logging.warning(message)
    
    @staticmethod
    def debug(message):
        logging.debug(message)
    
    @staticmethod
    def error(message):
        logging.error(message)
        