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
import logging
from typing import NoReturn, Optional


def info(msg: str) -> NoReturn:
    """
        Print information with "INFO" level.
    """
    print("[INFO]: {}".format(msg))


def test(msg: str) -> NoReturn:
    """
        Print information with "TEST" level.
    """
    print("[TEST]: {}".format(msg))


def warn(msg: str) -> NoReturn:
    """
        Print information with "WARN" level.
    """
    print("[WARNING]: {}".format(msg))


def error(msg: str) -> NoReturn:
    """
        Print information with "ERROR" level.
        And we exit from the entire program.
    """
    print("[ERROR]: {}".format(msg))
    exit(1)


def assert_error(msg: str) -> str:
	"""
		Print information with assertion.
	"""
	return "[ERROR]: {}".format(msg)


def if_exist(path: str, strict: bool = False) -> Optional[bool]:
    """
        Verify whether a given path exists or not. It will
        exit from the entire program if `strict` is set
        to False.
    """
    try:
        if os.path.exists(path):
            return True
        else:
            raise FileNotFoundError(path)
    except FileNotFoundError as e:
        warn("{} does not exist.".format(e))
        if not strict:
            return False
        else:
            exit(1)


def mkdir(path: str) -> NoReturn:
    """
        Create a directory.
    """
    if not if_exist(path):
        info("create directory: {}".format(path))
        os.makedirs(path, exist_ok=True)


def remove(path: str) -> NoReturn:
    """
        Remove a file or a directory.
    """
    if if_exist(path):
        if os.path.isfile(path):
            os.remove(path)
            info("remove %s" % path)
        elif os.path.isdir(path):
            if not os.listdir(path):
                # empty directory
                os.rmdir(path)
            else:
                shutil.rmtree(path)
            info("remove %s" % path)


def create_logger(log_file: str) -> logging.RootLogger:
    """
        Given a logging path, create a logger.
    """
    logging.basicConfig(
        filename=log_file,
        format="%(asctime)-15s: [%(levelname)s]: %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger
