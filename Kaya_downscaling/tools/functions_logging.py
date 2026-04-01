import logging
import atexit
from datetime import datetime
from pathlib import Path

from tools.general_functions import PRINT_COLORS, replace_punctuation_in_filenames

def cleanup_empty_logs(debug_logger, results_logger):
    # Close all handlers to release file locks
    for handler in debug_logger.handlers[:]:
        handler.close()
        debug_logger.removeHandler(handler)
    for handler in results_logger.handlers[:]:
        handler.close()
        results_logger.removeHandler(handler)

def init_logging(log_prefix='app', log_dir='log', console_level=logging.INFO,
                  results_to_console=False):
    """
    Set up dual logging system with debug and results logs.

    Args:
        log_prefix: Prefix for the log filenames
        log_dir: Directory to store logs (created if doesn't exist)
        console_level: Logging level for console output
        results_to_console: If True, results logger also prints to console

    Returns:
        tuple: (debug_logger, results_logger)
    """
    print(f"{PRINT_COLORS['cyan']}Initializing logging system with log directory: {log_dir}{PRINT_COLORS['end']}")
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create loggers
    debug_logger = logging.getLogger('debug')
    results_logger = logging.getLogger('results')

    # Clear existing handlers
    debug_logger.handlers.clear()
    results_logger.handlers.clear()

    # Set levels
    debug_logger.setLevel(logging.DEBUG)
    results_logger.setLevel(logging.INFO)

    # Create filenames with timestamps
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    debug_file = log_path / f'{log_prefix}_debug_{timestamp}.log'
    results_file = log_path / f'{log_prefix}_results_{timestamp}.log'

    # Create handlers
    debug_handler = logging.FileHandler(debug_file)
    results_handler = logging.FileHandler(results_file)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Create formatters with custom date format (no milliseconds)
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    results_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add formatters to handlers
    debug_handler.setFormatter(debug_formatter)
    results_handler.setFormatter(results_formatter)
    console_handler.setFormatter(debug_formatter)

    # Add handlers to loggers
    debug_logger.addHandler(debug_handler)
    debug_logger.addHandler(console_handler)
    results_logger.addHandler(results_handler)

    # Optionally add console output for results
    if results_to_console:
        results_console = logging.StreamHandler()
        results_console.setFormatter(results_formatter)
        results_logger.addHandler(results_console)

    # Prevent loggers from propagating to root logger
    debug_logger.propagate = False
    results_logger.propagate = False

    atexit.register(cleanup_empty_logs, debug_logger, results_logger)

    return debug_logger, results_logger
