import os, datetime
import fnmatch

def make_dir(new_folder: str) -> str:
    # Implement another try block if there are Permission problems
    try:
        os.makedirs(new_folder)
    except FileExistsError:
        pass
    return new_folder

def make_dir_timestamp(new_folder: str) -> str:
    date, time = str(datetime.datetime.now()).split()
    new_folder = new_folder + date[5:] + '-' + time[0:5]
    return make_dir(new_folder)

def find_pattern_in_path(path, pattern):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result