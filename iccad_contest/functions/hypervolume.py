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
from typing import List, NoReturn, Union

from iccad_contest.utils.basic_utils import error, assert_error
from iccad_contest.utils.constants import dim_of_objective_values


class HyperVolume(object):
    """
        NOTICE: We assume minimization in Pareto hypervolume computation.
        The implementation is refered to as:
            C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
            algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
            Computation, pages 1157-1163, Vancouver, Canada, July 2006.
    """

    def __init__(self, reference_point: List[float]):
        """
            Set the reference point for Pareto hypervolume computation.
        """
        self.reference_point = reference_point
        self.list = []

    @property
    def reference_point(self) -> List[float]:
        """
            Return the original reference point.
            For more information, please refer it to `reference_point.setter`.
        """
        _reference_point = self._reference_point.copy()
        _reference_point[0] = -_reference_point[0]
        return _reference_point

    @reference_point.setter
    def reference_point(self, reference_point: List[float]) -> None:
        """
            Set the reference point.
            NOTICE: our design space exploration problem is a maximization problem.
            However, our Pareto hypervolume computation is based on a minimization
            problem. So, we should negate our reference point in general.
            Specifically, we only need to negate the performance metric.
        """
        self._reference_point = reference_point.copy()
        self._reference_point[0] = -self._reference_point[0]

    def compute(self, predict_point: Union[np.ndarray, torch.Tensor]) -> float:
        """
            Compute the Pareto hypervolume that is dominated by a non-dominated
            `predict_point`. `predict_point` and `reference_point` are translated, so
            that the reference point is [0, ..., 0], before the Pareto
            hypervolume computation.
        """

        def weakly_dominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        """
            NOTICE: our design space exploration problem is a maximization problem. And
            We compute the Pareto hypervolume for the minimization problem.
            So, we internally flip the sign of the `predict_point` and the reference point.
            The reference point is automatically flipped using `self._reference_point`.
            Specifically, we only flip the performance metric value.
        """
        if isinstance(predict_point, (torch.Tensor, np.ndarray)):
            if isinstance(predict_point, torch.Tensor):
                predict_point = predict_point.clone()
            else:
                predict_point = predict_point.copy()
            predict_point[:, 0] = -predict_point[:, 0]
            predict_point = predict_point.tolist()
        elif isinstance(predict_point, list):
            if any(isinstance(point, list) for point in predict_point):
                """
                    Verify whether `predict_point` belongs to List[List[float]].
                """
                predict_point = np.array(predict_point)
                predict_point[:, 0] = -predict_point[:, 0]
                predict_point = predict_point.tolist()
            else:
                """
                    Verify whether `predict_point` belongs to List[float].
                """
                assert len(predict_point) == dim_of_objective_values, \
                    assert_error("unsupported predict point type: {} "
                        "in Pareto hypervolume computation.".format(
                            type(predict_point)
                        )
                    )
                predict_point = np.array(predict_point)
                predict_point[0] = -predict_point[0]
                predict_point = predict_point.tolist()
        else:
            error("unsupported predict point type: {} "
                "in Pareto hypervolume computation.".format(type(predict_point))
            )

        relevant_points = []
        reference_point = self._reference_point
        dimensions = len(reference_point)
        for point in predict_point:
            # only consider points that dominate the reference point
            if weakly_dominates(point, reference_point):
                relevant_points.append(point)
        if any(reference_point):
            # shift points so that `reference_point` == [0, ..., 0]
            # the `reference_point` have no need to be explicitly used
            # in the HV computation
            for j in range(len(relevant_points)):
                relevant_points[j] = [relevant_points[j][i] - reference_point[i] for i in range(dimensions)]
        self.pre_process(relevant_points)
        bounds = [-1.0e308] * dimensions
        hyper_volume = self.calc_hv(dimensions - 1, len(relevant_points), bounds)
        return hyper_volume

    def calc_hv(self, idx: int, length: int, bounds: List[float]) -> float:
        """
            In contrast to the paper, the implementation assumes that `reference_point` = [0, ..., 0], 
            which can reduce a few operations.
        """
        hyper_volume = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hyper_volume
        elif idx == 0:
            # NOTICE: only one dimension
            return -sentinel.next[0].cargo[0]
        elif idx == 1:
            # NOTICE: two dimensions
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                p_cargo = p.cargo
                hyper_volume += h * (q.cargo[1] - p_cargo[1])
                if p_cargo[0] < h:
                    h = p_cargo[0]
                q = p
                p = q.next[1]
            hyper_volume += h * q.cargo[1]
            return hyper_volume
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            calc_hv = self.calc_hv
            p = sentinel
            q = p.prev[idx]
            while q.cargo is not None:
                if q.ignore < idx:
                    q.ignore = 0
                q = q.prev[idx]
            q = p.prev[idx]
            while length > 1 and (
                    q.cargo[idx] > bounds[idx] or q.prev[idx].cargo[idx] >= bounds[idx]):
                p = q
                remove(p, idx, bounds)
                q = p.prev[idx]
                length -= 1
            q_area = q.area
            q_cargo = q.cargo
            q_prev_index = q.prev[idx]
            if length > 1:
                hyper_volume = q_prev_index.volume[idx] + q_prev_index.area[idx] * (
                    q_cargo[idx] - q_prev_index.cargo[idx]
                )
            else:
                q_area[0] = 1
                q_area[1:idx + 1] = [q_area[i] * -q_cargo[i] for i in range(idx)]
            q.volume[idx] = hyper_volume
            if q.ignore >= idx:
                q_area[idx] = q_prev_index.area[idx]
            else:
                q_area[idx] = calc_hv(idx - 1, length, bounds)
                if q_area[idx] <= q_prev_index.area[idx]:
                    q.ignore = idx
            while p is not sentinel:
                p_cargo_index = p.cargo[idx]
                hyper_volume += q.area[idx] * (p_cargo_index - q.cargo[idx])
                bounds[idx] = p_cargo_index
                reinsert(p, idx, bounds)
                length += 1
                q = p
                p = p.next[idx]
                q.volume[idx] = hyper_volume
                if q.ignore >= idx:
                    q.area[idx] = q.prev[idx].area[idx]
                else:
                    q.area[idx] = calc_hv(idx - 1, length, bounds)
                    if q.area[idx] <= q.prev[idx].area[idx]:
                        q.ignore = idx
            hyper_volume -= q.area[idx] * q.cargo[idx]
            return hyper_volume

    def pre_process(self, predict_point: List[List[float]]) -> NoReturn:
        """
            Sets up the multi-linked list data structure.
        """
        dimensions = len(self._reference_point)
        node_list = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in predict_point]
        for i in range(dimensions):
            self.sort_by_dimension(nodes, i)
            node_list.extend(nodes, i)
        self.list = node_list

    def sort_by_dimension(self, nodes: List, i: int) -> NoReturn:
        """
            Sorts the list of nodes by the i-th value of the contained points.
        """
        # build a list of tuples: (point[i], node)
        decorated = [(node.cargo[i], index, node) for index, node in enumerate(nodes)]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, _, node) in decorated]


class MultiList(object):
    """
        Multi-linked list consists of several doubly linked lists that share common nodes.
        every node has multiple predecessors and successors, one in every list.
    """

    class Node(object):
        """
            Node class specification.
        """

        def __init__(self, number_of_lists: int, cargo=None):
            """
                A base data structure for a node.
            """
            self.cargo: List[float] = cargo
            self.next = [None] * number_of_lists
            self.prev = [None] * number_of_lists
            self.ignore = 0
            self.area = [0.0] * number_of_lists
            self.volume = [0.0] * number_of_lists

        def __str__(self):
            """
                Stringify the point.
            """
            return str(self.cargo)

    def __init__(self, number_of_lists: int):
        """
            Builds `number_of_lists` doubly linked lists.
        """
        self.number_of_lists = number_of_lists
        self.sentinel = MultiList.Node(number_of_lists)
        self.sentinel.next = [self.sentinel] * number_of_lists
        self.sentinel.prev = [self.sentinel] * number_of_lists

    def __str__(self):
        """
            Stringify the multi-linked list.
        """
        strings = []
        for i in range(self.number_of_lists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        ret = ""
        for string in strings:
            ret += string + '\n'
        return ret

    def __len__(self):
        """
            Get the number of lists that are included in the `MultiList`
        """
        return self.number_of_lists

    def get_length(self, i: int) -> int:
        """
            Get the length of the i-th list.
        """
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index: int) -> NoReturn:
        """
            Appends a node to the end of the list at the given index.
        """
        last_but_one = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last_but_one
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last_but_one.next[index] = node

    def extend(self, nodes: List, index: int) -> NoReturn:
        """
            Extends the list at the given index with the nodes.
        """
        sentinel = self.sentinel
        for node in nodes:
            last_but_one = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = last_but_one
            # set the last element as the new one
            sentinel.prev[index] = node
            last_but_one.next[index] = node

    def remove(self, node, index: int, bounds: List[float]):
        """
            Removes and returns `node` from all lists in [0: `index`].
        """
        for i in range(index):
            prev = node.prev[i]
            succ = node.next[i]
            prev.next[i] = succ
            succ.prev[i] = prev
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index: int, bounds: List[float]) -> NoReturn:
        """
            Inserts `node` at the position it has in all lists in [0: `index`]
            before it was removed.
            The method assumes that the next and previous
            nodes of the node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
