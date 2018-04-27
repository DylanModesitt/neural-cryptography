# sys
import sys


class Logger:
    """
    A loger that, when set to sys.stdout logs both to the
    system stdout and also to a logfile specified as an input
    """
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass