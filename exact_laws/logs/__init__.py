import logging, sys

INFO = logging.INFO
WARNING = logging.WARNING
DEBUG = logging.DEBUG
ERROR = logging.ERROR


def getLogger(name=None):
    if name:
        return logging.getLogger("exact_law").getChild(name)
    return logging.getLogger("exact_law")


def setup(log_filename, log_level=logging.INFO):
    getLogger().setLevel(log_level)
    fh = logging.FileHandler(filename=log_filename)
    sh = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(module)-25s %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    getLogger().addHandler(fh)
    getLogger().addHandler(sh)
