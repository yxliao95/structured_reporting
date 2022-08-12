import pathspec  # https://python-path-specification.readthedocs.io/en/latest/readme.html
from pathspec import PathSpec
import os
import logging
from hydra import compose

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

        with open(ignore_source_path, "r") as f:
            ignore_pattern = "".join(f.readlines())
            self.ignore_matcher = pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, ignore_pattern.splitlines()
            )

    def ignore(self, filename):
        """ Return True if the input file should be ignored. """
        if not self.ignore_matcher:
            self._init_ignore_checker()
        return self.ignore_matcher.match_file(filename)
