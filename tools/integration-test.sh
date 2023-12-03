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

#!/bin/bash

set -ex
set -o pipefail


tools=`dirname ${BASH_SOURCE[0]}`
root=${tools}/..


function set_env() {
    function handler() {
        exit 1
    }
    trap 'handler' SIGINT
}


function integration_test() {

	function step_1() {
		# basic integration tests
		python3 ${root}/example_optimizer/random-search-optimizer.py \
			-o output-of-step-1 \
			-u "00ef538e88634ddd9810d034b748c24d" \
			-q 10
	}

	function step_2() {
		python3 ${root}/example_optimizer/random-search-optimizer.py \
			-o output-of-step-2 \
			-s ${root}/example_optimizer/configs.json \
			-u "00ef538e88634ddd9810d034b748c24d" \
			-q 10
	}

	function step_3() {
		python3 ${root}/example_optimizer/lr-offline-optimizer.py \
			-o output-of-step-3 \
			-u "00ef538e88634ddd9810d034b748c24d" \
			-q 10
	}

	function step_4() {
		python3 ${root}/example_optimizer/lr-online-optimizer.py \
			-o output-of-step-4 \
			-u "00ef538e88634ddd9810d034b748c24d" \
			-q 10
	}

	function step_5() {
		python3 ${root}/example_optimizer/gp-optimizer.py \
			-o output-of-step-5 \
			-s ${root}/example_optimizer/configs.json \
			-u "00ef538e88634ddd9810d034b748c24d" \
			-s ${root}/example_optimizer/gp-configs.json \
			-q 10
	}

	step_1
	step_2
	step_3
	step_4
	step_5
	echo "[INFO]: integration test is passed."
}


function main() {
	set_env
	integration_test
}


main
