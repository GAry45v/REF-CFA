import logging
import sys

class Logger:
    """
    Singleton-like wrapper for the standard Python logging facility.
    Ensures consistent formatting across the distributed system.
    """
    _instance = None

    @staticmethod
    def get_logger(name="RefCFA"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
        return logger