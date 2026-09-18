from functools import wraps
from time import perf_counter


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            elapsed_seconds = perf_counter() - start_time

            hours, remainder = divmod(int(elapsed_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(
                f"{func.__name__}() completed in "
                f"{hours:02d} hours, "
                f"{minutes:02d} minutes, "
                f"{seconds:02d} seconds"
            )

    return wrapper
