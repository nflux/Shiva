from datetime import datetime
from loguru import logger
import os

class Logger(object):
    def __init__(self):
        now = datetime.now()
        current_time = now.strftime("%m-%d-%Y  %H:%M:%S")
        folder = os.path.join("./logs", str(now.year), now.strftime("%B"))
        self.check_create_directory(folder)
        filename = os.path.join(folder, str(now.day) + '.log')
        logger.add(filename, enqueue=True, backtrace=True, diagnose=True)

    def info(self, msg, to_print=False):
        logger.debug(msg)
        if to_print:
            print(msg)

    def exception(self, msg):
        logger.exception(msg)

    def close(self):
        logger.info("Closing logs")
        logger.info("******************************************************************************")

    def check_create_directory(self, _dir):
        '''
            Creates a directory if doesn't exist
        '''
        if not os.path.exists(_dir):
            os.makedirs(_dir)