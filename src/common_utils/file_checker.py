import logging
import os

import pathspec  # https://python-path-specification.readthedocs.io/en/latest/readme.html
from pathspec import PathSpec

logger = logging.getLogger()

pkg = os.path.dirname(__file__)
src = os.path.dirname(pkg)
ignore_source_path = os.path.join(src, "config", "ignore", "common_ignore")


class FileChecker:
    def __init__(self):
        self.ignore_matcher: PathSpec = None

    def _init_ignore_checker(self):
        if not os.path.exists(ignore_source_path):
            logger.error("Not found %s.", ignore_source_path)
            raise Exception(f"Not found {ignore_source_path}")

        with open(ignore_source_path, "r", encoding="UTF-8") as f:
            ignore_pattern = "".join(f.readlines())
            self.ignore_matcher = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, ignore_pattern.splitlines()
            )

    def ignore(self, filename):
        """ Skip the hidden folder (e.g. .DS_Store in MacOS). 
        Return True if the input file should be ignored.
        """
        if not self.ignore_matcher:
            self._init_ignore_checker()
        return self.ignore_matcher.match_file(filename)

    def filter(self, file_list):
        """ Removed unwanted files from the list and return a new list. """
        new_file_list = []
        for file in file_list:
            if not self.ignore(file):
                new_file_list.append(file)
        return new_file_list
