# Author: baichen.bai@alibaba-inc.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import _io
from typing import NoReturn, Tuple
from pathvalidate.argparse import validate_filename, validate_filepath
from iccad_contest.utils.basic_utils import mkdir, warn, assert_error


def validate_file_path(path: str) -> NoReturn:
    """
        Verifying whether the file_path is a valid file path or not
        for a target platform. If the `platform` is set to "auto",
        the caller will detect the platform automatically.
    """
    validate_filepath(path, platform="auto")


def validate_file_name(file: str) -> NoReturn:
    """
        Verify whether a file name is a valid name.
    """
    validate_filename(file, platform="universal")


def open_with_abs_path(path: str, mode: str) -> _io.TextIOWrapper:
    """
        Open a file with a specified mode.
    """
    assert os.path.isabs(path), \
        assert_error(
            "an absolute path is expected, but the platform gets {}".format(path)
        )
    return open(path, mode)


def validate_abs_path(path: str, strict: bool = False) -> str:
    """
        Return the absolute path of `path`.
        Verify whether the `path` does exist or not if `strict`
        is set to True.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if strict:
        assert os.path.isdir(path), \
            assert_error("{} does not exist.".format(path))
    return path


def _path_join(*args: Tuple[str]) -> str:
    """
        Join a set of strings as a file path.
    """
    assert len(args) > 1
    path, file = args[:-1], args[-1]
    path = validate_abs_path(os.path.join(*path), strict=False)
    assert os.path.basename(file) == file, \
        assert_error("a file name is expected, but the platform gets {}".format(file))
    return os.path.join(path, file)


def _strict_path_join(*args: Tuple[str]) -> str:
    """
        Join a set of strings as a file path strictly.
    """
    assert len(args) > 1
    path, file = args[:-1], args[-1]
    path = validate_abs_path(os.path.join(*path), strict=True)
    assert os.path.basename(file) == file, \
        assert_error("a file name is expected, but the platform gets {}".format(file))
    return os.path.join(path, file)


def path_join_for_read(*args: Tuple[str]) -> str:
    """
        Join a set of strings as a file path to read.
    """
    file = _strict_path_join(*args)
    assert os.path.isfile(file), \
        assert_error("a file is expected, but the platform gets {}".format(file))
    return file


def path_join_for_write(*args: Tuple[str]) -> str:
    """
        Join a set of strings as a file path to write.
    """
    file = _strict_path_join(*args)
    if os.path.isfile(file):
        warn("{} already exists.".format(file))
    return file

def path_join_for_write_with_create(*args: Tuple[str]) -> str:
    """
        Join a set of strings as a path. If the path
        is a directory path and does not exist, we
        create the directory path.
    """
    file = _path_join(*args)
    if not os.path.isfile(file):
        mkdir(os.path.dirname(file))
    return file
