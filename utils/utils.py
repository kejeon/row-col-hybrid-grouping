import os
import time
from functools import wraps
import socket
from datetime import datetime
import pytz
from pytz import timezone
import logging
import yaml

# Global variables that need to be initialized
tz = None       # init with set_timezone
logger = None   # init with create_logger

def set_timezone(my_timezone: str = 'Asia/Seoul') -> None:
    """Sets the global variable for timezone and 
    creates a custom converter for logging.Formatter to use the given timezone. 
    If a given timezone is not found, it uses UTC.

    Call this function before calling any other functions from this module.

    Parameters
    ----------
    my_timezone : str, optional
        Timezone to be configured, by default 'Asia/Seoul'.

    Raises
    ------
    pytz.exceptions.UnknownTimeZoneError
        If the given timezone is not found.

    Examples
    --------
    Known timezone
    >>> set_timezone('Asia/Seoul')

    Unknown timezone
    >>> set_timezone('USB/Eastern')
    Unknown timzone. Using UTC.
    """
    global tz
    try:
        tz = timezone(my_timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        print('Unknown timzone. Using UTC.')
        tz = timezone('UTC')

    def timetz(*args):
        return datetime.now(tz).timetuple()
    
    logging.Formatter.converter = timetz

    return

def timeit(func):
    """Decorator function to measure the time taken by a function. 
    The measured time is logged using the logger of the decorated function.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Decorated function.

    Examples
    --------
    >>> @timeit
    ... def my_function():
    ...     time.sleep(1)
    ...     return
    >>> my_function()
    1994-06-01 91:06:21 - my_function - INFO - Execution time: 1.0000 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = logging.getLogger()
        logger.info(f"Execution time: {end_time-start_time:.4f} seconds")
        return result
    return wrapper

def setup_output_dir(output_dir: str) -> None:
    """Setup the output directory with the given path. 
    Check if the directory exists, if not create it.

    Parameters
    ----------
    output_dir : str
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else:
    #     raise FileExistsError(f"Directory {output_dir} already exists.")
    
    return

def setup_log_dir(log_dir:str) -> str:
    """Setup the log directory with the given path. 
    Create a run directory under the given log directory and return the run directory path.

    Parameters
    ----------
    log_dir : str
        Path to log directory

    Returns
    -------
    run_dir : str
        Path to run directory.

    Examples
    --------
    >>> run_dir = setup_log_dir('./logs')
    >>> print(run_dir)
    logs/run_2021-08-21_12-34-56
    """
    if tz == None:
        set_timezone('UTC')

    run_dir_name = f"run_{datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join(log_dir, run_dir_name)
    setup_output_dir(log_dir)
    return log_dir


def setup_dir(dir_path:str) -> str:
    if tz == None:
        set_timezone('UTC')

    dir_path_name = f"{dir_path}_{datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')}"
    # log_dir = os.path.join(log_dir, run_dir_name)
    setup_output_dir(dir_path_name)
    return dir_path_name

def create_logger(logger_name:str,
                  log_dir:str
                  ) -> logging.Logger:
    """Create and return a logger with the given name and log directory.
    Stream handler is set to INFO level and File handler is set to DEBUG level.

    Parameters
    ----------
    logger_name : str
    log_dir : str

    Returns
    -------
    logging.Logger
    
    Examples
    --------
    Basic usage.
    >>> logger = create_logger('my_logger', './logs')
    >>> logger.info('This is a INFO message')
    >>> logger.debug('This is a DEBUG message')
    1994-06-01 91:06:21 - my_logger - INFO - This is a INFO message

    Advanced usage.
    >>> set_timezone('USA/Eastern')
    >>> log_dir = setup_log_dir('./logs')
    >>> logger = create_logger('my_logger', log_dir)
    >>> logger.debug('This is a DEBUG message')
    1994-06-01 91:06:21 - my_logger - INFO - This is a INFO message
    """
    global logger
    global tz
    if logger is not None:
        logger_info(f"Logger {logger_name} already exists.")
        return logger

    logger = logging.getLogger(logger_name)
    logger.propagate = False    
    # stream_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                                      datefmt='%Y-%m-%d %H:%M:%S')
    # file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
    #                                    datefmt='%Y-%m-%d %H:%M:%S')
    stream_formatter = logging.Formatter(fmt='%(message)s',
                                         datefmt='%Y-%m-%d %H:%M:%S')
    file_formatter = logging.Formatter(fmt='%(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')

    log_file_name = f"log_{logger_name}_{socket.gethostname()}.ans"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.DEBUG)

    return logger

def save_dict(my_dict, filename):
    with open(filename, 'w') as f:
        yaml.dump(my_dict, f)

def load_dict(filename):
    with open(filename, 'r') as f:
        content = f.read()
        my_dict = yaml.load(content, Loader=yaml.FullLoader)
    return my_dict

# ANSI escape codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
UNDERLINE = "\033[4m"
ITALIC = "\033[3m"
BOLD = "\033[1m"
RESET = "\033[0m"
INDENT = "   "

# logger macros
def logger_info(msg):
    if logger == None:
        return
    logger.info(f"{msg}")

def logger_bold(msg):
    logger_info(f"{BOLD}{msg}{RESET}")

def logger_lvln(msg, num_indent=1):
    logger_info(f"{INDENT*num_indent}{msg}")

def logger_lvl1(msg):
    logger_lvln(msg, 1)

def logger_lvl2(msg):
    logger_lvln(msg, 2)
    # logger.info(f"{LEVEL2}{msg}{RESET}")

def logger_lvl3(msg):
    logger_lvln(msg, 3)
    # logger.info(f"{LEVEL2}{msg}{RESET}")

def logger_newline():
    logger_info(f"")

def green(msg):
    return f"{GREEN}{msg}{RESET}"

def green_bold(msg):
    return f"{GREEN}{BOLD}{msg}{RESET}"


