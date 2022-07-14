import os

def get_data_dir_name():
    return "data_blobs"

def get_root_path(path=""):
    return os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))

def get_data_root_path(path="", try_mount_path=False):
    data_root_path = os.path.join(get_root_path(), os.path.join(get_data_dir_name(), path))
    return data_root_path


