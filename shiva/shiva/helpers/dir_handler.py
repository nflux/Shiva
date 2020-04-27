import os, datetime
import fnmatch

def make_dir(new_folder: str, use_existing=False) -> str:
    # Implement another try block if there are Permission problems
    if use_existing:
        try:
            os.makedirs(new_folder)
        except FileExistsError:
            pass
    else:
        i = 1
        temp = new_folder
        while os.path.isdir(temp):
            temp = new_folder + '_' + str(i)
            i += 1
        new_folder = temp
        os.makedirs(new_folder)
    return new_folder

def make_dir_timestamp(new_folder: str, create_new_timestamp_folder=False, name_append: str=None, use_existing=False) -> str:
    date, time = str(datetime.datetime.now()).split()
    tmpst = date[5:] + '-' + time[0:5]
    if name_append is not None:
        tmpst = name_append + '-' + tmpst
    if create_new_timestamp_folder:
        new_folder = os.path.join(new_folder, tmpst)
    else:
        new_folder = new_folder + '-' + tmpst
    new_folder = new_folder.replace(':', '')
    return make_dir(new_folder, use_existing=use_existing)

def find_pattern_in_path(path, pattern):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # if fnmatch.fnmatch(name, pattern):
            if pattern in name:
                result.append(os.path.join(root, name))
    return result