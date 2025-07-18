import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Adds a new logging level to the `logging` module and the configured logging class.

    `levelName` becomes an attribute of `logging` with the value `levelNum`. 
    `methodName` becomes a convenience method for `logging` and `logging.getLoggerClass()`. 
    If `methodName` is not provided, `levelName.lower()` is used.

    Example:
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel('TRACE')
    >>> logging.getLogger(__name__).trace('Test message')
    >>> logging.TRACE
    5
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError(f'{levelName} already exists in logging module')
    if hasattr(logging, methodName):
        raise AttributeError(f'{methodName} already exists in logging module')
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError(f'{methodName} already exists in logger class')

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def setup_logging():
    """
    Configures logging for `agile` desktop automation.
    Supports log levels: 'debug', 'info', and 'result'.
    """

    try:
        addLoggingLevel('RESULT', 35)
    except AttributeError:
        pass  # Level already exists

    log_type = os.getenv('AGILE_LOGGING_LEVEL', 'info').lower()

    if logging.getLogger().hasHandlers():
        return

    root = logging.getLogger()
    root.handlers = []  # Clear existing handlers

    class AgileFormatter(logging.Formatter):
        """Custom logging formatter for agile automation"""

        def format(self, record):
            if isinstance(record.name, str) and record.name.startswith('agile.'):
                record.name = record.name.split('.')[-2]
            return super().format(record)

    # Console logging setup
    console = logging.StreamHandler(sys.stdout)

    if log_type == 'result':
        console.setLevel('RESULT')
        console.setFormatter(AgileFormatter('%(message)s'))
    else:
        console.setFormatter(AgileFormatter('%(levelname)-8s [%(name)s] %(message)s'))

    root.addHandler(console)

    # Switch-case for log levels
    if log_type == 'result':
        root.setLevel('RESULT')
    elif log_type == 'debug':
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # Configure `agile` logger
    agile_logger = logging.getLogger('agile')
    agile_logger.propagate = False
    agile_logger.addHandler(console)
    agile_logger.setLevel(root.level)

    logger = logging.getLogger('agile')
    logger.info(f'Agile logging setup complete with level {log_type}')

    # Silence third-party loggers
    for logger_name in [
        'WDM',
        'httpx',
        'urllib3',
        'asyncio',
        'langchain',
        'openai',
        'httpcore',
        'charset_normalizer',
        'PIL.PngImagePlugin',
    ]:
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False
