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
    Linear regression (online) optimizer constructs a linear regression model
    for the design space exploration.
    It updates the model progressively and picks the most valuable design to evaluate,
    hoping to reduce the total running time.
    A command to test "lr-offline-optimizer.py":
    ``` 
        python3 lr-online-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    You can specify more options to test your optimizer. please use
    ```
        python3 lr-online-optimizer.py -h
    ```
    to check the help menu.
    The code is only for example demonstration.
"""


import torch
import random
import sklearn
import numpy as np
from typing import List, NoReturn
from sklearn.linear_model import LinearRegression

from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier
from iccad_contest.functions.design_space import MicroarchitectureDesignSpace


class OnlineLinearRegressionOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space: MicroarchitectureDesignSpace):
        """
            Build a wrapper class for an optimizer.
        """
        AbstractOptimizer.__init__(self, design_space)
        self.model = LinearRegression()
        self.n_suggestions = 5
        self.x = []
        self.y = []

    def suggest(self)  -> List[List[int]]:
        """
            Get a suggestion from the optimizer.
            The method returns next guesses. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
        """
        x_guess = random.sample(
            range(1, self.design_space.size + 1),
            k=self.n_suggestions
        )
        potential_suggest =  [
            self.design_space.vec_to_microarchitecture_embedding(
                self.design_space.idx_to_vec(_x_guess)
            ) for _x_guess in x_guess
        ]
        try:
            """
                NOTICE: we can also use the model to sweep the design space if 
                the design space is not quite large.
                We only use a very naive way to pick up the design just for demonstration only.
            """
            ppa = torch.Tensor(self.model.predict(np.array(potential_suggest)))
            potential_parteo_frontier = get_pareto_frontier(ppa)
            _potential_suggest = []
            for point in potential_parteo_frontier:
                index = torch.all(ppa == point.unsqueeze(0), axis=1)
                _potential_suggest.append(
                    torch.Tensor(potential_suggest)[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            return _potential_suggest
        except sklearn.exceptions.NotFittedError:
            return potential_suggest

    def observe(self, x: List[List[int]], y: List[List[float]]) -> NoReturn:
        """
            Send an observation of a suggestion back to the optimizer.
            `x` is the output of `suggest`. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
            `y` is the corresponding PPA values (where each `x` is mapped to).
        """
        for _x in x:
            self.x.append(_x)
        for _y in y:
            self.y.append(_y)
        self.model.fit(np.array(self.x), np.array(self.y))


"""
    The main function.
"""
if __name__ == "__main__":
    # please specifiy `experiment` as the main function entry point
    experiment(OnlineLinearRegressionOptimizer)
