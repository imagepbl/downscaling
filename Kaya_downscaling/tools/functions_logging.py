import inspect
import logging
import os
from datetime import datetime
from pathlib import Path

from tools.general_functions import PRINT_COLORS

class CallerFilter(logging.Filter):
    """Attach %(caller)s: the function that called the function which
    emitted the log record."""

    def filter(self, record):
        record.caller = "unknown"
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back  # leave filter() itself
            # Skip the logging machinery to reach the emitting function
            # (normcase needed: logging._srcfile is normcased, co_filename is not,
            # so a raw comparison fails on Windows due to path casing)
            while (frame is not None
                   and os.path.normcase(frame.f_code.co_filename) == logging._srcfile):
                frame = frame.f_back
            # One frame further up is the caller of the emitting function
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                record.caller = (f"{Path(caller.f_code.co_filename).stem}:"
                                 f"{caller.f_code.co_name}:{caller.f_lineno}")
        return True


def init_logging(log_prefix="app", log_dir="log", console_level=logging.INFO,
                 results_to_console=False):
    """
    Set up a three-file logging system: debug, results, and errors.

    Creates timestamped log files in log_dir:
        - log_debug_<prefix>_<timestamp>.log: all messages (DEBUG and above)
          from the debug logger, also mirrored to the console
        - log_results_<prefix>_<timestamp>.log: messages (INFO and above)
          from the results logger
        - log_errors_warning_<prefix>_<timestamp>.log: WARNING and above from
          both loggers, plus warnings.warn() calls (e.g. from rasterio, xarray,
          rioxarray, dask) captured via logging.captureWarnings(True)

    Debug, console, and error output include the emit location
    (module:function:line) and the caller of the emitting function
    via the CallerFilter.

    Note:
        Calling this function enables logging.captureWarnings(True) globally,
        so warnings are redirected from stderr to the "py.warnings" logger
        and appear in the error log and on the console. Re-running the
        function closes and replaces all previously attached handlers.

    Args:
        log_prefix: Prefix for the log filenames
        log_dir: Directory to store logs (created if it doesn't exist)
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
    debug_logger = logging.getLogger("debug")
    results_logger = logging.getLogger("results")

    # Clear existing handlers
    for handler in debug_logger.handlers[:]:
        handler.close()
        debug_logger.removeHandler(handler)
    for handler in results_logger.handlers[:]:
        handler.close()
        results_logger.removeHandler(handler)

    # Set levels
    debug_logger.setLevel(logging.DEBUG)
    results_logger.setLevel(logging.INFO)

    # Create filenames with timestamps
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    debug_file = log_path / f"log_debug_{log_prefix}_{timestamp}.log"
    results_file = log_path / f"log_results_{log_prefix}_{timestamp}.log"

    # Create handlers
    debug_handler = logging.FileHandler(debug_file)
    results_handler = logging.FileHandler(results_file)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Create formatters with custom date format (no milliseconds)
    debug_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s:%(funcName)s:%(lineno)d "
        "(from %(caller)s) - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    results_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Add formatters to handlers
    debug_handler.setFormatter(debug_formatter)
    results_handler.setFormatter(results_formatter)
    console_handler.setFormatter(debug_formatter)

    # Error/warning log file (captures WARNING and above)
    error_file = log_path / f"log_errors_warning_{log_prefix}_{timestamp}.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.WARNING)
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(module)s:%(funcName)s:%(lineno)d "
        "(from %(caller)s) - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_formatter)

    # Attach the caller filter to every handler whose formatter uses %(caller)s
    caller_filter = CallerFilter()
    debug_handler.addFilter(caller_filter)
    console_handler.addFilter(caller_filter)
    error_handler.addFilter(caller_filter)

    debug_logger.addHandler(error_handler)
    results_logger.addHandler(error_handler)

    # Route warnings.warn() calls (rasterio, xarray, rioxarray, dask) into the same file
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    for handler in warnings_logger.handlers[:]:
        handler.close()
        warnings_logger.removeHandler(handler)
    warnings_logger.addHandler(error_handler)
    warnings_logger.addHandler(console_handler)
    warnings_logger.propagate = False

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

    return debug_logger, results_logger
