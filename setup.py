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
from pathlib import Path
from setuptools import find_packages, setup


PLATFORM_NAME = "iccad-contest"


# load the long description from `README.md`
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def read_requirements(name):
    with open(os.path.join("requirements", "{}.in".format(name))) as f:
        requirements = f.read().strip()
    requirements = requirements.replace("==", ">=").splitlines()
    return [p for p in requirements if p[0].isalnum()]


# derive install requirements from base.in requirements.
requirements = read_requirements("base")
opt_requirements = read_requirements("optimizers")
ipynb_requirements = read_requirements("ipynb")


setup(
    name=PLATFORM_NAME,
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "iccad_contest.contest-dataset": [
            "*.csv",
            "*.xlsx"
        ]
    },
    url="https://code.aone.alibaba-inc.com/ctl-research/iccad-contest-platform",
    author="Chen BAI",
    author_email=("baichen.bai@alibaba-inc.com"),
    license="Apache v2",
    description="ICCAD'22 Contest @ Microarchitecture Design Space Exploration",
    install_requires=requirements,
    extras_require={
        "optimizers": opt_requirements,
        "notebooks": ipynb_requirements
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["any"]
)
