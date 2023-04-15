import os
import shutil


def remove_dirs(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)


def check_and_remove_dirs(dir_path: str, do_remove: bool):
    """ Remove the input dir if the dir exist """
    if do_remove and os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def check_and_create_dirs(dir_path: str):
    """ Create dir if it does not exist """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def check_and_remove_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def check_and_move_file(file_source_path:str, file_destination_path:str):
    check_and_create_dirs(os.path.dirname(file_destination_path))
    shutil.move(file_source_path, file_destination_path)