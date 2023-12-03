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


import numpy as np
import pandas as pd
from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import List, Optional, NoReturn
from iccad_contest.functions.dataset import load_contest_design_space
from iccad_contest.utils.basic_utils import warn, error, assert_error


class DesignSpace(ABC):
    """
        Base class of the design space.
    """

    def __init__(
        self,
        size: int,
        dims_of_index_vector: int,
        dims_of_microarchitecture_embedding: int
    ):
        """
            size: total size of the design space
            dims_of_index_vector:
                dimension of a microarchitecture embedding index vector
            dims_of_microarchitecture_embedding:
                dimension of a microarchitecture embedding
        """
        self.size = size
        self.dims_of_index_vector = dims_of_index_vector
        self.dims_of_microarchitecture_embedding = dims_of_microarchitecture_embedding

    @abstractmethod
    def contain(self, microarchitecture_embeddings: List[int]) -> NoReturn:
        """
            Verify whether a given microarchitecture embedding is valid.
        """
        raise NotImplementedError

    @abstractmethod
    def idx_to_vec(self, idx: int) -> None:
        """
            Translate a given index to a corresponding vector.
            idx: the index of a microarchitecture
        """
        raise NotImplementedError

    @abstractmethod
    def vec_to_idx(self, vec: List[int]) -> None:
        """
            Translate a given vector to a corresponding index.
            vec: the vector of a microarchitecture encoding
        """
        raise NotImplementedError

    @abstractmethod
    def vec_to_microarchitecture_embedding(self, vec: List[int]) -> None:
        """
            Translate a given vector to a corresponding
            microarchitecture embedding.
        """
        raise NotImplementedError


class MicroarchitectureDesignSpace(DesignSpace):
    """
        Derived class of the microarchitecture design space.
    """

    def __init__(self,
        descriptions: OrderedDict, components_mappings: OrderedDict, size: int
    ):
        """
            descriptions: <class "collections.OrderedDict">
            Example:
            descriptions = {
                "sub-design-1": {
                    "Fetch": [1],
                    "Decoder": [1],
                    "ISU": [1, 2, 3],
                    "IFU": [1, 2, 3],
                    ...
                }
            }

            components_mappings: <class "collections.OrderedDict">
            Example:
            components_mappings = {
                "Fetch": {
                    "description": ["FetchWidth"],
                    "1": [4]
                },
                "Decoder": {
                    "description": ["DecodeWidth"],
                    "1": [1]
                },
                "ISU": {
                    "description": [
                        "MEM_INST.DispatchWidth", "MEM_INST.IssueWidth"
                        "MEM_INST.NumEntries", "INT_INST.DispatchWidth",
                        "INT_INST.IssueWidth", "INT_INST.NumEntries",
                        "FP_INST.DispatchWidth", "FP_INST.IssueWidth",
                        "FP_INST.NumEntries"
                    ],
                    "1": [1, 1, 8, 1, 1, 8, 1, 1, 8],
                    "2": [1, 1, 6, 1, 1, 6, 1, 1, 6],
                    "3": [1, 1, 10, 1, 1, 12, 1, 1, 12]
                },
                "IFU": {
                    "description": ["BranchTag", "FetchBufferEntries", "FetchTargetQueue"]
                    "1": [8, 8, 16],
                    "2": [6, 6, 14],
                    "3": [10, 12, 20]
                },
                ...
            }

            size: <int> the size of the entire design space
        """
        self.descriptions = descriptions
        self.components_mappings = components_mappings
        # construct look-up tables
        # self.designs: <list>
        # Example:
        #   ["sub-design-1", ...]
        self.designs = list(self.descriptions.keys())
        # self.components: <list>
        # Example:
        #   ["Fetch", "Decoder", "ISU", ...]
        self.components = list(self.descriptions[self.designs[0]].keys())
        self.component_offset = self.construct_component_offset()
        self.design_size = self.construct_design_size()
        self.acc_design_size = list(map(
                lambda x, idx: np.sum(x[:idx]),
                [self.design_size for i in range(len(self.design_size))],
                range(1, len(self.design_size) + 1)
            )
        )
        self.component_dims = self.construct_component_dims()
        super(MicroarchitectureDesignSpace, self).__init__(
            size,
            len(self.components),
            sum([
                    len(self.components_mappings[c]["description"]) \
                        for c in self.components
                ]
            )
        )

    def construct_component_offset(self) -> List[int]:
        """
            Construct component offset look-up table.
        """
        component_offset = []
        for k, v in self.descriptions.items():
            _component_offset = []
            for _k, _v in v.items():
                _component_offset.append(_v[0])
            component_offset.append(_component_offset)
        return component_offset

    def construct_design_size(self) -> List[int]:
        """
            Construct design space size look-up table.
        """
        design_size = []
        for k, v in self.descriptions.items():
            _design_size = []
            for _k, _v in v.items():
                _design_size.append(len(_v))
            design_size.append(np.prod(_design_size))
        return design_size

    def construct_component_dims(self) -> List[int]:
        """
            Construct component dimensions look-up table.
        """
        component_dims = []
        for k, v in self.descriptions.items():
            _component_dims = []
            for _k, _v in v.items():
                _component_dims.append(len(_v))
            component_dims.append(_component_dims)
        return component_dims

    def idx_to_vec(self, idx: int) -> List[int]:
        """
            Translate a given index to a corresponding vector.
            idx: the index of a microarchitecture
        """
        idx -= 1
        assert idx >= 0, assert_error("invalid index.")
        assert idx < self.size, assert_error("index exceeds the size of design space.")
        vec = []
        design = np.where(np.array(self.acc_design_size) > idx)[0][0]
        if design >= 1:
            # NOTICE: subtract the offset
            idx -= self.acc_design_size[design - 1]
        for dim in self.component_dims[design]:
            vec.append(idx % dim)
            idx //= dim
        # add the offset
        for i in range(len(vec)):
            vec[i] += self.component_offset[design][i]
        return vec

    def vec_to_idx(self, vec: List[int]) -> int:
        """
            Translate a given vector to a corresponding index.
            vec: the vector of a microarchitecture encoding
        """
        def _get_design() -> Optional[int]:
            # a tricky to identify the design
            for k, v in self.descriptions.items():
                if v["Decoder"][0] == vec[1]:
                    return v["Decoder"][0] - 1
            error("an invalid index vector.")

        idx = 0
        design = _get_design()
        # subtract the offset
        for i in range(len(vec)):
            vec[i] -= self.component_offset[design][i]
        for j, k in enumerate(vec):
            idx += int(np.prod(self.component_dims[design][:j])) * k
        if design >= 1:
            # NOTICE: add the offset
            idx += self.acc_design_size[design - 1]
        assert idx >= 0, assert_error("invalid index.")
        assert idx < self.size, assert_error("index exceeds the size of design space.")
        idx += 1
        return idx

    def get_mapping_params(self, vec: List[int], idx: int) -> int:
        """
            Given a vector and an offset index, get a candidate value
            for a target component.
        """
        return self.components_mappings[self.components[idx]][vec[idx]]

    def vec_to_microarchitecture_embedding(self, vec: List[int]) -> List[int]:
        """
            Translate a given vector to a corresponding
            microarchitecture embedding.
        """
        microarchitecture_embedding = []
        for i in range(len(vec)):
            for param in self.get_mapping_params(vec, i):
                microarchitecture_embedding.append(param)
        return microarchitecture_embedding

    def contain(self, microarchitecture_embeddings: List[int]) -> NoReturn:
        """
            Verify whether a given microarchitecture embedding is valid.
        """
        def _get_design(vec: List[int]) -> int:
            # a tricky to identify the design
            for k, v in self.descriptions.items():
                if v["Decoder"][0] == vec[1]:
                    return self.designs[v["Decoder"][0] - 1]
            raise ValueError

        for microarchitecture_embedding in microarchitecture_embeddings:
            vec = microarchitecture_embedding
            try:
                design = _get_design(vec)
            except ValueError:
                raise ValueError
            try:
                offset = 0
                for i in range(len(self.components)):
                    exist = False
                    l = len(self.components_mappings[self.components[i]]["description"])
                    for k, v in self.components_mappings[self.components[i]].items():
                        if vec[offset : offset + l] == v:
                            exist = True
                    if not exist:
                        warn(
                            "component: {} - {} of the suggest " \
                            "microarchitecture embedding: {} is invalid.".format(
                                self.components[i],
                                vec[offset : offset + l],
                                microarchitecture_embedding
                            )
                        )
                        raise ValueError
                    offset += l
            except Exception:
                raise ValueError


def parse_microarchitecture_design_space_sheet(
    microarchitecture_design_space_sheet: pd.core.frame.DataFrame
) -> OrderedDict:
    """
        Parse the microarchitecture design space sheet.
    """
    descriptions = OrderedDict()
    head = microarchitecture_design_space_sheet.columns.tolist()

    # parse design space
    for row in microarchitecture_design_space_sheet.values:
        # extract designs
        descriptions[row[0]] = OrderedDict()
        # extract components
        for col in range(1, len(head) - 1):
            descriptions[row[0]][head[col]] = []
        # extract candidate values
        for col in range(1, len(head) - 1):
            try:
                # multiple candidate values
                for item in list(map(lambda x: int(x), row[col].split(','))):
                    descriptions[row[0]][head[col]].append(item)
            except AttributeError:
                # single candidate value
                descriptions[row[0]][head[col]].append(row[col])
    return descriptions


def parse_components_sheet(components_sheet: pd.core.frame.DataFrame) -> OrderedDict:
    """
        Parse the components design space sheet.
    """
    components_mappings = OrderedDict()
    head = components_sheet.columns.tolist()

    # construct look-up tables
    # mappings: <list> [name, width, idx]
    # Example:
    #   mappings = [("ISU", 10, 0), ("IFU", 4, 10), ("ROB", 2, 14)]
    mappings = []
    for i in range(len(head)):
        if not head[i].startswith("Unnamed"):
            if i == 0:
                name, width, idx = head[i], 1, i
            else:
                mappings.append((name, width, idx))
                name, width, idx = head[i], 1, i
        else:
            width += 1
    mappings.append((name, width, idx))

    for name, width, idx in mappings:
        # extract components
        components_mappings[name] = OrderedDict()
        # extract descriptions
        components_mappings[name]["description"] = []
        for i in range(idx + 1, idx + width):
            components_mappings[name]["description"].append(components_sheet.values[0][i])
        # extract candidate values
        # get number of rows, a trick to test <class "float"> of nan
        nrow = np.where(components_sheet[name].values == \
            components_sheet[name].values)[0][-1]
        for i in range(1, nrow + 1):
            components_mappings[name][int(i)] = \
                list(components_sheet.values[i][idx + 1: idx + width])
    return components_mappings


def parse_contest_design_space() -> MicroarchitectureDesignSpace:
    """
        Parse the contest design space.
    """
    microarchitecture_design_space_sheet, components_sheet = load_contest_design_space()
    descriptions = parse_microarchitecture_design_space_sheet(
        microarchitecture_design_space_sheet
    )
    components_mappings = parse_components_sheet(
        components_sheet
    )

    return MicroarchitectureDesignSpace(
        descriptions,
        components_mappings,
        int(microarchitecture_design_space_sheet.values[0][-1])
    )
