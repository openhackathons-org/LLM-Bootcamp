# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


Bootstrap: docker
From: nvcr.io/nvidia/nemo:dev 

%environment
    export XDG_RUNTIME_DIR=

%post
    
    pip install ujson
    pip install --upgrade --no-cache-dir gdown   
    pip install jupyterlab datasets
    pip install rouge-score
	pip install evaluate
	pip install tqdm
	pip install huggingface_hub[hf_xet]
    
%runscript
    "$@"

%labels
    Author Tosin
    
