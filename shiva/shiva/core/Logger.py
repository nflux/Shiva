from datetime import datetime
import logging
import os

class Logger(object):
    def __init__(self):
        """
        This class wraps standard Python logging functions (https://docs.python.org/3/library/logging.html#module-logging)
        All Shiva components instantiates a copy of this class for their logging purposes.
        """
        now = datetime.now()
        current_time = now.strftime("%m-%d-%Y  %H:%M:%S")
        folder = os.path.join("./logs", str(now.year), now.strftime("%B"))
        self.check_create_directory(folder)
        filename = os.path.join(folder, str(now.day) + '.log')
        logging.basicConfig(filename=filename, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
        logging.info("**************************** " + current_time + " ****************************")

    def info(self, msg, to_print=False):
        """
        Log a info type message as https://docs.python.org/3/library/logging.html#logging.Logger.info

        Args:
            msg (str): string message to be logged
            to_print (bool): optionally output message to terminal. Default False.

        Returns:
            None
        """
        logging.info(msg)
        if to_print:
            print(msg)

    def error(self, msg):
        logging.error(msg, exc_info=True)

    def close(self):
        """
        Close log file

        Returns:
            None
        """
        logging.info("Closing logs")
        logging.info("******************************************************************************")

    def check_create_directory(self, _dir):
        '''
            Creates a directory if doesn't exist
        '''
        if not os.path.exists(_dir):
            os.makedirs(_dir)