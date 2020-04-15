#coding:utf-8

import os
import json
import logging
import os
import linecache
import time
import datetime
from functools import wraps
from logging.handlers import TimedRotatingFileHandler


class Logger:
    """
    logger module
    """
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def get_logger(self, logger_name='job_logger'):
        """
        logger recorder object
        :return: logger recorder object
        """
        # first, create logger item
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            # handle to log file
            logfile = self.log_path
            fh = TimedRotatingFileHandler(filename=logfile, when='D', interval=1, backupCount=10000)
            fh.setLevel(logging.INFO)

            # handle to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # output format
            formatter = logging.Formatter("%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # add logger to handle
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger

