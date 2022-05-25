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


