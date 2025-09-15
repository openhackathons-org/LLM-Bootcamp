# Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.

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
    
