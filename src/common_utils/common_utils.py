import os
import shutil


def remove_all(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
