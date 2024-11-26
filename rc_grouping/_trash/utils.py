import os
import time
from functools import wraps, partial
import socket
from datetime import datetime
import pytz
from pytz import timezone
import logging

# Global variable for timezone. Initialize this first. 
tz = None

def setup_output_dir(output_dir: str) -> None:
    """Setup the output directory with the given path. 
    Check if the directory exists, if not create it.

    Parameters
    ----------
    output_dir : str
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return

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
    try:
        tz = timezone(my_timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        print('Unknown timzone. Using UTC.')
        tz = timezone('UTC')

    def timetz(*args):
        return datetime.now(tz).timetuple()
    
    logging.Formatter.converter = timetz

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
    logger = logging.getLogger(logger_name)    
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                         datefmt='%Y-%m-%d %H:%M:%S')
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')

    log_file_name = f"log_{logger_name}_{socket.gethostname()}.txt"
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

# def _pseudo_timeit(func, logger=None):
#     """Decorator function to measure the time taken by a function. 
#     The measured time is logged using the logger of the decorated function.

#     Parameters
#     ----------
#     func : function
#         Function to be decorated.

#     Returns
#     -------
#     wrapper : function
#         Decorated function.

#     Examples
#     --------
#     >>> @timeit
#     ... def my_function():
#     ...     time.sleep(1)
#     ...     return
#     >>> my_function()
#     1994-06-01 91:06:21 - my_function - INFO - Execution time: 1.0000 seconds
#     """
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         if logger is None:
#             logger = logging.getLogger()
#         logger.info(f"Execution time: {end_time-start_time:.4f} seconds")
#         return result
#     return wrapper

def timeit(logger=None):
    def inner(func, logger = logger):
        def wrapper(logger = logger, *args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if logger is None:
                logger = logging.getLogger()
            logger.info(f"Execution time: {end_time-start_time:.4f} seconds")
            return result
        return wrapper
    return inner