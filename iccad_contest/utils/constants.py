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
from iccad_contest.utils.path_utils import path_join_for_read


# solution configuration specification
solution_settings = "configs.json"


# contest dataset root specification
root_name_of_contest_dataset = "contest-dataset"


# contest design space specification
name_of_contest_design_space = "design-space.xlsx"
contest_design_space = path_join_for_read(
    os.path.dirname(__file__),
    os.path.pardir,
    root_name_of_contest_dataset,
    name_of_contest_design_space
)


# contest dataset specification
name_of_contest_dataset = "contest.csv"
contest_dataset = path_join_for_read(
    os.path.dirname(__file__),
    os.path.pardir,
    root_name_of_contest_dataset,
    name_of_contest_dataset
)


# output file name specification
format_of_exp_output_root = "iccad-contest-%Y%m%d-%H%M%S"
derived_root = "derived"
log_root = "log"
evaluation_root = "evaluation"
time_root = "time"
explored_microarchitecture_embedding_root = "explored-microarchitecture-embedding"
list_of_exp_output_sub_root = [
    derived_root,
    log_root,
    evaluation_root,
    time_root,
    explored_microarchitecture_embedding_root
]
suffix_of_log = ".log"


# summary report specification
summary_root = "summary"
suffix_of_summary = ".rpt"


# objectives specification
# performance, power, and area
dim_of_objective_values = 3
