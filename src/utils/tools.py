import time
from pathlib import Path

def list_dirs(base_path):
    subject_dirs = []
    for p in Path(base_path).iterdir():
        if p.is_dir():
            subject_dirs.append(p)
    return subject_dirs

def list_files(dir):
    return [x for x in dir.glob("**/*") if x.is_file()]

def glob_file(path, g):
    return [x for x in path.glob(g) if x.is_file()][0]

def timeit(method):
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()

        print(f"Elapsed time for {method.__name__}: {end -start}")
        return result
    return timed

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer Class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it!")

        self._start_time = time.perf_counter()
    
    def stop(self, msg=None):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it!")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if msg is not None:
            print(f"Elapsed time for {msg}: {elapsed_time:0.4f} seconds")
        else:
            print(f"Elapsed time: {elapsed_time:0.4f} seconds")
