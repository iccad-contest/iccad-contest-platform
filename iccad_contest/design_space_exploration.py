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
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "utils")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "functions")
)
import copy
import torch
import random
import logging
import numpy as np
from time import time
from collections import OrderedDict
from typing import Dict, NoReturn, Tuple, List

from iccad_contest.__init__ import __version__
from iccad_contest.utils.constants import summary_root
from iccad_contest.utils.numpy_utils import random_seed
from iccad_contest.functions.hypervolume import HyperVolume
from iccad_contest.utils.serialize import ContestSerializer
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.utils.arguments import experiment_parser, parse_args, \
    load_solution_settings
from iccad_contest.functions.design_space import MicroarchitectureDesignSpace
from iccad_contest.utils.basic_utils import create_logger, error, assert_error
from iccad_contest.functions.problem import PPAT, DesignSpaceExplorationProblem, get_pareto_frontier


logger = None


def initialize_logger(output_path: str, uuid: str) -> logging.RootLogger:
    """
        Given a specified output path and a UUID, create a logger.
    """
    global logger
    log_file = ContestSerializer.logging_path(output_path, uuid)
    logger = create_logger(log_file)
    return logger


def create_problem() -> DesignSpaceExplorationProblem:
    """
        Create the contest design space exploration problem.
    """
    return DesignSpaceExplorationProblem()


def set_random_seed(seed: str) -> NoReturn:
    """
        Given a hexidecimal UUID string, set the random seed.
    """
    random_seed_generator = random.Random(int(seed, 16))
    random.seed(random_seed(random_seed_generator))
    np.random.seed(random_seed(random_seed_generator))
    logger.info("use UUID {} to reproduce the experiment results.".format(seed))


def do_suggest(
    iteration: int,
    design_space: MicroarchitectureDesignSpace,
    optimizer: AbstractOptimizer,
    suggest_time: List[float]
) -> List[List[int]]:
    """
        Suggest the samples to evaluate.
    """
    t = time()
    try:
        next_microarchitecture_embeddings = optimizer.suggest()
        if next_microarchitecture_embeddings is None:
            raise Exception
    except Exception as e:
        logger.warning(
            "failed in 'suggest' of the optimizer. " \
            "the error message: {}" \
            "\nfall back to random search.".format(e)
        )
        random_index = random.sample(
            range(design_space.size),
            k=1
        )
        next_microarchitecture_embeddings = [
            design_space.vec_to_microarchitecture_embedding(
                design_space.idx_to_vec(_random_index)
            ) for _random_index in random_index
        ]
    suggest_time.append(time() - t)
    logger.info(
        "{}-th query, " \
        "next microarchitecture embeddings: {} " \
        "'suggest' time: {}.".format(
            iteration + 1,
            str(next_microarchitecture_embeddings),
            suggest_time[iteration]
        )
    )

    try:
        design_space.contain(next_microarchitecture_embeddings)
    except Exception:
        raise ValueError(
            "suggestions provided by the optimizer are out of the design space."
        )

    return next_microarchitecture_embeddings


def do_evaluate(
    iteration: int,
    problem: DesignSpaceExplorationProblem,
    next_microarchitecture_embeddings: List[List[int]],
    explored_microarchitecture_embeddings: List[List[int]],
    objective_values: List[List[float]],
    evaluation_time: List[float]
) -> NoReturn:
    """
        Evaluate samples suggested by the optimizer.
    """
    for i, next_microarchitecture_embedding \
        in enumerate(next_microarchitecture_embeddings):
        t = time()
        try:
            ppat = problem.evaluate(next_microarchitecture_embedding)
        except Exception as e:
            logger.warning(
                "failed in evaluating microarchitecture embeddings. " \
                "set the infinity as the evaluated results."
            )
            ppat = PPAT(0, float("inf"), float("inf"), float("inf"))
        evaluation_time.append(time() - t + ppat.time_of_vlsi_flow)
        explored_microarchitecture_embeddings.append(next_microarchitecture_embedding)
        objective_values.append(ppat.get_objective_values())
        logger.info(
            "evaluate {}-th candidate in a batch, " \
            "microarchitecture embedding: {}, " \
            "normalized performance: {}, " \
            "normalized power: {}, " \
            "normalized area: {}, " \
            "time of the VLSI flow: {}.".format(
                i + 1,
                next_microarchitecture_embedding,
                ppat.performance,
                ppat.power,
                ppat.area,
                ppat.time_of_vlsi_flow
            )
        )


def do_observe(
    iteration: int,
    optimizer: AbstractOptimizer,
    next_microarchitecture_embeddings: List[List[int]],
    objective_values: List[List[float]],
    observe_time: List[float]
):
    """
        Return evaluated samples to the optimizer.
    """
    t = time()
    try:
        optimizer.observe(
            next_microarchitecture_embeddings,
            copy.deepcopy(objective_values[-len(next_microarchitecture_embeddings):])
        )
    except Exception as e:
        logger.warning(
            "failed in 'observe' of the optimizer. " \
            "the error message: {}" \
            "\nignore observations.".format(e)
        )
        logger.exception(e, exc_info=True)
    observe_time.append(time() - t)


def do_experiment_impl(
    problem: DesignSpaceExplorationProblem,
    optimizer: AbstractOptimizer,
    num_of_queries: int
) -> Tuple[
        Tuple[List[List[int]], List[List[float]], List[List[float]]], \
        Tuple[List[float], List[float]], List[float]
    ]:
    """
        Main function implementation of the experiment.
    """
    suggest_time = []
    evaluation_time = []
    observe_time = []
    explored_microarchitecture_embeddings = []
    objective_values = []
    hv = HyperVolume(problem.reference_point)

    for i in range(num_of_queries):
        next_microarchitecture_embeddings = do_suggest(
            i,
            problem.design_space,
            optimizer,
            suggest_time
        )
        do_evaluate(
            i,
            problem,
            next_microarchitecture_embeddings,
            explored_microarchitecture_embeddings,
            objective_values,
            evaluation_time
        )
        do_observe(
            i,
            optimizer,
            next_microarchitecture_embeddings,
            objective_values,
            observe_time
        )
        curr_hypervolume = hv.compute(
            get_pareto_frontier(
                torch.Tensor(objective_values)
            )
        )
        logger.info(
            "summary for {}-th query, " \
            "current Pareto hypervolume: {}, " \
            "cost: {}.".format(
                i + 1,
                curr_hypervolume,
                np.sum([
                        np.sum(suggest_time),
                        np.sum(evaluation_time),
                        np.sum(observe_time)
                    ]
                )
            )
        )
        if optimizer.early_stopping:
            break

    curr_hypervolume = hv.compute(
        get_pareto_frontier(
            torch.Tensor(objective_values)
        )
    )
    logger.info(
        "summary for the solution, " \
        "the best Pareto hypervolume: {}, " \
        "cost: {}.".format(
            curr_hypervolume,
            np.sum([
                    np.sum(suggest_time),
                    np.sum(evaluation_time),
                    np.sum(observe_time)
                ]
            )
        )
    )

    return (explored_microarchitecture_embeddings, objective_values, curr_hypervolume), \
        (suggest_time, evaluation_time, observe_time)


def experiment_impl(args: Dict, solution: AbstractOptimizer, solution_settings: Dict) -> \
    Tuple[
        Tuple[List[List[int]], List[List[float]], List[List[float]]], \
        Tuple[List[float], List[float]], List[float]
    ]:
    """
        Main function entry point of the experiment.
    """
    problem = create_problem()
    optimizer = solution(problem.design_space, **solution_settings)
    return do_experiment_impl(
        problem,
        optimizer,
        args["num_of_queries"]
    )


def parse_experiment_args() -> Dict:
    """
        Parse the arguments.
    """
    args = parse_args(
        experiment_parser(
            "ICCAD'22 Contest Platform - solutions evaluation"
        )
    )
    return args


def save_experiment(
    args: Dict,
    solution_settings: Dict,
    explored_result: Tuple[List[List[int]], List[List[float]], List[List[float]]],
    timing: Tuple[List[float], List[float], List[float]]
) -> NoReturn:
    """
        Experimental results saving implementation.
    """
    save_dict = OrderedDict()
    _args = args.copy()
    _args["uuid"] = _args["uuid"].hex
    save_dict["command"] = _args
    save_dict["solution-settings"] = solution_settings
    save_dict["explored-microarchitecture-embedding"] = explored_result[0]
    save_dict["objective-values"] = explored_result[1]
    save_dict["Pareto hypervolume"] = explored_result[2]
    save_dict["suggest-time"] = timing[0]
    save_dict["evaluate-time"] = timing[1]
    save_dict["observe-time"] = timing[2]
    save_dict["total-time"] = np.sum([
            np.sum(timing[0]),
            np.sum(timing[1]),
            np.sum(timing[2])
        ]
    )
    save_path = ContestSerializer.save(args["output_path"], args["uuid"], save_dict)
    logger.info(
        "save the experiment result. " \
        "please refer them to {}".format(
            save_path
        )
    )


def experiment(solution: AbstractOptimizer):
    """
        The main function for ICCAD'22 contests.
    """
    args = parse_experiment_args()
    logger = initialize_logger(args["output_path"], args["uuid"])
    logger.info("Prolem C @ ICCAD'22 contest platform version: {}".format(__version__))
    solution_settings = load_solution_settings(args["solution_settings"])
    if solution_settings == {}:
        logger.info("no --solution-settings is specified in arguments.")
    else:
        logger.info("load solution settings: {}".format(solution_settings))
    set_random_seed(args["uuid"].hex)
    explored_result, timing = experiment_impl(args, solution, solution_settings)
    save_experiment(args, solution_settings, explored_result, timing)
