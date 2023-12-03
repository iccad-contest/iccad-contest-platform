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


from abc import ABC, abstractmethod
from importlib_metadata import version
from typing import Dict, List, NoReturn

from iccad_contest.functions.design_space import MicroarchitectureDesignSpace


class AbstractOptimizer(ABC):
    """
        Abstract base class for the optimizers.
        it creates a common API across all packages.
    """

    # every implementation package needs to specify `primary_import`,
    # e.g.,
    # primary_import = opentuner
    primary_import = None

    def __init__(self, design_space: MicroarchitectureDesignSpace, **kwargs: Dict):
        """
            Build a wrapper class for an optimizer.
        """
        self.design_space = design_space
        # early stopping criterion
        self.early_stopping = False

    @classmethod
    def get_version(cls) -> str:
        """
            Get the version for the optimizer. Usually, the version number of an
            optimizer is equivalent to `package.__version__`.
        """
        assert (cls.primary_import is None) or isinstance(cls.primary_import, str)
        # verstion "x.x.x" is used as a default version
        version_str = "x.x.x" \
            if cls.primary_import is None else version(cls.primary_import)
        return version_str

    @abstractmethod
    def suggest(self) -> List[List[int]]:
        """
            Get a suggestion from the optimizer.
            The method returns next guesses. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self, x: List[List[int]], y: List[List[float]]) -> NoReturn:
        """
            Send an observation of a suggestion back to the optimizer.
            `x` is the output of `suggest`. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
            `y` is the corresponding PPA values (where each `x` is mapped to).
        """
        raise NotImplementedError
