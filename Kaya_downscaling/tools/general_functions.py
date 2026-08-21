import math
from datetime import datetime
from time import time
from typing import Any

# https://www.geeksforgeeks.org/python/print-colors-python-terminal/
PRINT_COLORS = {
    "end":          "\033[0m",
    "red":          "\033[91m",
    "green":        "\033[92m",
    "yellow":       "\033[93m",
    "blue":         "\033[94m",
    "purple":       "\033[95m",
    "cyan":         "\033[96m",
    "light_grey":   "\033[97m",
    "orange":       "\033[38;2;255;165;0m",
}

def calc_hours_minutes_seconds(start: datetime, end: datetime) -> tuple:
    delta = end - start
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    return hours, minutes, seconds

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"{PRINT_COLORS["green"]}Function {func.__name__!r} executed in {(t2-t1):.4f}s{PRINT_COLORS["end"]}")
        return result
    return wrap_func

def replace_punctuation_in_filenames(s:str) -> str:
    return s.replace("|","_").replace(" ","_").replace(",","").replace(".","").replace("(","").replace(")","")

def round_to_half(n):
    return round(n * 2) / 2

def is_int_or_half(n):
    return math.isclose(n % 1, 0) or math.isclose(n % 1, 0.5)

def format_factor(factor):
    """Return factor as int if it is a whole number, else rounded to two decimals."""
    return int(factor) if factor.is_integer() else round(factor, 2)

# used to use a flexible data directory root in settings_data_locations.json
def apply_root_json(obj, root) -> Any:
    if isinstance(obj, dict):
        return {k: apply_root_json(v, root) for k, v in obj.items()}
    if isinstance(obj, str) and obj.startswith("{data_root}/"):
        return root.rstrip("/") + "/" + obj[len("{data_root}/"):]
    return obj
