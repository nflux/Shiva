import os, datetime
from typing import List


def make_dir(new_folder: str, use_existing: bool = False) -> str:
    """
    Creates a new folder in the file system.
    Uses `os.makedirs()`. For more details https://docs.python.org/3/library/os.html#os.makedirs

    Args:
        new_folder (str): folder name. Can be absolute directory paths.
        use_existing (bool): If True and a folder already exists, will pass. If False and a folder exists, the new folder will have a numerical extention to make it unique.

    Returns:
        str: name of the new folder created
    """
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


def make_dir_timestamp(new_folder: str, create_new_timestamp_folder: bool = False, name_append: str= None, use_existing: bool = False) -> str:
    """
    This function creates a folder with a timestamp extension to the `new_folder` name.

    Args:
        new_folder (str): folder name. Can be absolute directory paths.
        create_new_timestamp_folder: If True, a timestamp folder will be created. If False, the timestamp string will be added to the end of the `new_folder` name
        name_append: Optionally add a name extension to the `new_folder` name
        use_existing: If True and a folder already exists, will pass. If False and a folder exists, the new folder will have a numerical extention to make it unique.

    Returns:
        str: name of the new folder created
    """
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


def find_pattern_in_path(path, pattern) -> List[str]:
    """
    This function searches for `pattern` in the given `path`.
    Primarily used to load agents from a given directory or checkpoint.

    Args:
        path: path string where to look for the `pattern`
        pattern: string pattern to be searched in the path

    Returns:
        List[str]: list of found patterns

    Example:
        >>> find_pattern_in_path('Control-Tasks', 'MPIEnv')
        ['./Control-Tasks/docs/source/rst/envs/MPIEnv.rst', './Control-Tasks/shiva/shiva/envs/MPIEnv.py']

    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # if fnmatch.fnmatch(name, pattern):
            if pattern in name:
                result.append(os.path.join(root, name))
    return result
