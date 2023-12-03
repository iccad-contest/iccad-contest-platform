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
import numpy as np
import pandas as pd
from typing import Tuple, Union, NoReturn

from iccad_contest.utils.basic_utils import if_exist, info, warn, assert_error
from iccad_contest.utils.constants import contest_design_space, contest_dataset


def load_txt(path: str, fmt=int) -> np.ndarray:
    """
        Load the contest data set from a given path.
    """
    if if_exist(path):
        info("loading from {}".format(path))
        return np.loadtxt(path, dtype=fmt)
    else:
        warn("cannot load {}".format(path))


def load_excel(path: str, sheet_name: Union[int, str] = 0) -> pd.core.frame.DataFrame:
    """
        Load an excel file from a given path. And read the
        sheet according to the given sheet name.
    """
    if_exist(path, strict=True)
    data = pd.read_excel(path, sheet_name=sheet_name)
    info("read the sheet {} of an Excel file from {}".format(sheet_name, path))
    return data


def load_contest_dataset() -> np.ndarray:
    """
        Load the contest data set.
    """
    def validate_contest_dataset(dataset: np.ndarray) -> NoReturn:
        assert dataset.ndim == 2, \
            assert_error(
                "{} dimensions are expected, but the platform gets {}.".format(dataset.ndim)
            )
        assert np.all(np.isfinite(dataset)), \
            assert_error(
                "dataset contains infinity."
            )
    dataset = load_txt(contest_dataset, fmt=float)
    validate_contest_dataset(dataset)
    return dataset


def load_contest_design_space() -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    """
        Load the contest design space from a fixed path.
    """
    microarchitecture_design_space_sheet = load_excel(
        contest_design_space,
        sheet_name="Microarchitecture Design Space"
    )
    components_sheet = load_excel(
        contest_design_space,
        sheet_name="Components"
    )
    return microarchitecture_design_space_sheet, components_sheet
