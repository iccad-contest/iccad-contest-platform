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
    Linear regression (offline) optimizer constructs a linear regression model
    for the design space exploration.
    It trains the model with many selected samples and freezes the model to search.
    A command to test "lr-offline-optimizer.py":
    ``` 
        python3 lr-offline-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    Set the '--num-of-queries' to 2 to avoid more time cost.
    You can specify more options to test your optimizer. please use
    ```
        python3 lr-offline-optimizer.py -h
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
from sklearn.metrics import mean_absolute_percentage_error

from iccad_contest.design_space_exploration import experiment
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.functions.problem import get_pareto_frontier
from iccad_contest.functions.design_space import MicroarchitectureDesignSpace


class OfflineLinearRegressionOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space: MicroarchitectureDesignSpace):
        """
            Build a wrapper class for an optimizer.
        """
        AbstractOptimizer.__init__(self, design_space)
        self.model = LinearRegression()
        """
            NOTICE: you can put `self.initial_size`, `self.training_size`, etc.,
            to a JSON file for better coding and tuning experience.
        """
        self.initial_size = 50
        self.training_size = round(0.8 * self.initial_size)
        self.n_suggestions = 10
        self.fit = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()

    def construct_microarchitecture_embedding_set(self) -> np.ndarray:
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)
                )
            )
        return np.array(microarchitecture_embedding_set)

    def suggest(self)  -> List[List[int]]:
        """
            Get a suggestion from the optimizer.
            The method returns next guesses. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
        """
        try:
            """
                NOTICE: we can also use the model to sweep the design space if 
                the design space is not quite large.
                We only use a very naive way to pick up the design just for demonstration only.
            """
            ppa = torch.Tensor(self.model.predict(self.microarchitecture_embedding_set))
            potential_parteo_frontier = get_pareto_frontier(ppa)
            potential_suggest = []
            for point in potential_parteo_frontier:
                potential_suggest.append(
                    self.microarchitecture_embedding_set[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            return potential_suggest
        except sklearn.exceptions.NotFittedError:
            x_guess = random.sample(
                range(1, self.design_space.size + 1), k=self.initial_size
            )
            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            return potential_suggest

    def observe(self, x: List[List[int]], y: List[List[float]]) -> NoReturn:
        """
            Send an observation of a suggestion back to the optimizer.
            `x` is the output of `suggest`. That is, a vector of a vector of integers,
            representing a series of microarchitecture embeddings.
            `y` is the corresponding PPA values (where each `x` is mapped to).
        """
        if self.fit:
            """
                NOTICE: we can check the model's accuracy and verify if it is
                suitable for the exploration.
            """
            total_x = np.array(x)
            total_y = np.array(y)
            training_x = total_x[:self.training_size + 1]
            training_y = total_y[:self.training_size + 1]
            testing_x = total_x[self.training_size:]
            testing_y = total_y[self.training_size:]
            self.model.fit(training_x, training_y)
            pred = self.model.predict(testing_x)
            mape = mean_absolute_percentage_error(testing_y, pred)
            print(
                "linear regression trained on data size: {}, " \
                "mean absolute percentage error: {}".format(
                    self.training_size,
                    mape
                )
            )
            self.fit = False


"""
    The main function.
"""
if __name__ == "__main__":
    # please specifiy `experiment` as the main function entry point
    experiment(OfflineLinearRegressionOptimizer)
