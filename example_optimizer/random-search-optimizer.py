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


"""
    Random search optimizer performs random search.
    The code is only for example demonstration.
    ``` 
        python3 random-search-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    You can specify more options to test your optimizer. please use
    ```
        python3 random-search-optimizer.py -h
    ```
    to check the help menu.
"""


import numpy as np
from typing import List, NoReturn

from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.design_space import MicroarchitectureDesignSpace

class RandomSearchOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space: MicroarchitectureDesignSpace):
        """
            Build a wrapper class for an optimizer.
        """
        AbstractOptimizer.__init__(self, design_space)
        self.n_suggestions = 1

    def suggest(self)  -> List[List[int]]:
        """
            Get a suggestion from the optimizer.
            The method returns next guesses. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
        """
        x_guess = np.random.choice(
            range(1, self.design_space.size + 1),
            size=self.n_suggestions
        )
        return [
            self.design_space.vec_to_microarchitecture_embedding(
                self.design_space.idx_to_vec(_x_guess)
            ) for _x_guess in x_guess
        ]

    def observe(self, x: List[List[int]], y: List[List[float]]) -> NoReturn:
        """
            Send an observation of a suggestion back to the optimizer.
            `x` is the output of `suggest`. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
            `y` is the corresponding PPA values (where each `x` is mapped to).
        """
        pass


"""
    The main function.
"""
if __name__ == "__main__":
    # please specifiy `experiment` as the main function entry point
    experiment(RandomSearchOptimizer)
