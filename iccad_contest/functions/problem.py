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


import torch
import numpy as np
from scipy import stats
from typing import List, NoReturn
from itertools import combinations
from abc import ABC, abstractmethod


from iccad_contest.utils.basic_utils import assert_error
from iccad_contest.functions.dataset import load_contest_dataset
from iccad_contest.utils.constants import dim_of_objective_values
from iccad_contest.functions.design_space import MicroarchitectureDesignSpace, \
    parse_contest_design_space


class Problem(ABC):
    """
        Base class for a problem.
    """

    def __init__(self):
        """
            The problem base class manages the data structure
            of the design space.
        """
        super(Problem, self).__init__()
        self._design_space = None

    @property
    def design_space(self) -> MicroarchitectureDesignSpace:
        """
            Get the design space.
        """
        return self._design_space

    @design_space.setter
    def design_space(self, design_space: MicroarchitectureDesignSpace) -> NoReturn:
        """
            Set the design space.
        """
        self._design_space = design_space

    @design_space.deleter
    def design_space(self) -> NoReturn:
        """
            Delete the design space.
        """
        del self._design_space

    @abstractmethod
    def evaluate(self, points: List[int]):
        """
            Evaluate the given microarchitecture embedding.
        """
        raise NotImplementedError


class PPAT(object):
    """
        Wrapped class for PPA values and the time of the VLSI flow.
    """
    def __init__(
        self, performance: float, power: float, area: float, time_of_vlsi_flow: float
    ):
        super(PPAT, self).__init__()
        self._performance = performance
        self._power = power
        self._area = area
        self._time_of_vlsi_flow = time_of_vlsi_flow

    @property
    def performance(self) -> float:
        """
            Get the performance value.
        """
        return self._performance

    @property
    def power(self) -> float:
        """
            Get the power value.
        """
        return self._power

    @property
    def area(self) -> float:
        """
            Get the area value.
        """
        return self._area

    @property
    def time_of_vlsi_flow(self) -> float:
        """
            Get the time of the VLSI flow.
        """
        return self._time_of_vlsi_flow

    def get_objective_values(self) -> List[float]:
        """
            Get the PPA values.
        """
        return [
            self.performance, self.power, self.area
        ]


class DesignSpaceExplorationProblem(Problem):
    """
        Derived class of the design space exploration problem.
        In the problem, our target is try to maximize the
        performance and minimize the power and area.
    """

    def __init__(self):
        super(DesignSpaceExplorationProblem, self).__init__()
        self.design_space: MicroarchitectureDesignSpace = parse_contest_design_space()
        self.dataset: np.ndarray = load_contest_dataset()
        self.dim_of_objective_values: int = dim_of_objective_values
        self.microarchitecture_embedding_set: np.ndarray = self.dataset[
            :, :-(dim_of_objective_values + 1)
        ]
        self.ppa: np.ndarray = stats.zscore(self.dataset[:, -4:-1])
        self.performance: np.ndarray = self.ppa[:, -3]
        self.power: np.ndarray = self.ppa[:, -2]
        self.area: np.ndarray = self.ppa[:, -1]
        self.time_of_vlsi_flow: np.ndarray = self.dataset[:, -1]
        self.pareto_frontier: torch.Tensor = self.get_pareto_frontier()
        self.reference_point: List[float] = self.calc_reference_point()

    def evaluate(self, microarchitecture_embedding: List[int]) -> PPAT:
        """
            Given a microarchitecture embedding, return PPA values.
        """
        idx = [
            all(x) \
                for x in np.equal(
                    self.microarchitecture_embedding_set,
                    microarchitecture_embedding
                )
        ].index(True)
        return PPAT(
            self.performance[idx],
            self.power[idx],
            self.area[idx],
            self.time_of_vlsi_flow[idx]
        )

    def get_pareto_frontier(self) -> torch.Tensor:
        """
            Get the golden Pareto frontier from the data set.
        """
        return get_pareto_frontier(torch.Tensor(self.ppa))

    def calc_reference_point(self) -> List[float]:
        """
            For the definition of the reference point, each element
            of the reference point should be the worst.
        """

        """
            NOTICE: The worst value of each metric
            (i.e., performance, power, and area) is defined as the
            110% *de facto* worst value in the data set. So, we should
            obtain the absolute value for a metric
            (i.e., positive and negative). Then, we get the
            maximal value among them. Finally, we scale the value with
            the "1.1" or "-1.1" coefficient.
        """
        return [
            -1.1 * np.max([abs(np.min(self.performance)), abs(np.max(self.performance))]),
            1.1 * np.max([abs(np.min(self.power)), abs(np.max(self.power))]),
            1.1 * np.max([abs(np.min(self.area)), abs(np.max(self.area))])
        ]


def _get_non_dominated(Y: torch.Tensor, maximize: bool = True) -> torch.Tensor:
    """
        Get the non dominated mask. We leverage maximization implementation.
        Please refer it to `get_non_dominated` for more information.
    """
    is_efficient = torch.ones(
        *Y.shape[:-1],
        dtype=bool,
        device=Y.device
    )
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            update[..., i] = i_is_efficient.clone()
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                is_efficient2[~i_is_efficient] = False
            is_efficient[is_efficient2] = update[is_efficient2]
    return is_efficient


def get_non_dominated(Y: torch.Tensor, deduplicate: bool = True) -> torch.Tensor:
    """
        Get the non dominated mask. We leverage maximization implementation.
    """
    MAX_BYTES = 5e6
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(
            Y.shape[:-1],
            dtype=torch.bool,
            device=Y.device
        )
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or \
        n ** 2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _get_non_dominated(Y)

    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask


def get_pareto_frontier(objective_values: torch.Tensor) -> torch.Tensor:
    """
        NOTICE: `get_pareto_frontier` assumes maximization.
    """
    assert isinstance(objective_values, torch.Tensor), \
        assert_error("please convert the input to 'torch.Tensor.")

    _objective_values = objective_values.clone()

    """
        NOTICE: Since our problem is try to maximize the performance
        and minimize the power and area. So, we negate the power and
        area values in view of the maximization implementation of
        `get_non_dominated`.
    """
    for i in [1, 2]:
        _objective_values[:, i] = -_objective_values[:, i]
    return objective_values[get_non_dominated(_objective_values)]


def get_adrs(
    reference_pareto_frontier: torch.Tensor,
    predict_pareto_frontier: torch.Tensor
) -> float:
    """
        DEPRECATED method.
    """
    # calculate average distance to the `reference_pareto_frontier` set
    assert isinstance(reference_pareto_frontier, torch.Tensor) and \
        isinstance(reference_pareto_frontier, torch.Tensor), \
            assert_error("please convert the input to 'torch.Tensor.")
    adrs = 0
    reference_pareto_frontier = reference_pareto_frontier.cpu()
    predict_pareto_frontier = predict_pareto_frontier.cpu()
    for omega in reference_pareto_frontier:
        mini = float('inf')
        for gamma in predict_pareto_frontier:
            mini = min(mini, np.linalg.norm(omega - gamma))
        adrs += mini
    adrs = adrs / len(reference_pareto_frontier)
    return adrs
