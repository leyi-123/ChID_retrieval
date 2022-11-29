import logging


def highlight(raw_str):
    return '\033[32m ' + raw_str + ' \033[0m'


class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
        formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y/%m/%d %H:%M:%S')
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def get_logger(self):
        self.logger.info(highlight(f'Logging to {self.logfile}!'))
        return self.logger

# if __name__ == '__main__':
#     logger = Logger('tmp.txt').get_logger()
#     logger.info('AFSAFASD')
