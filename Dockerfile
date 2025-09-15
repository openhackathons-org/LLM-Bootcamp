# Copyright (c) 2025 NVIDIA Corporation.  All rights reserved.

# Select Base Image 
FROM nvcr.io/nvidia/nemo:dev


# Pip install.

RUN pip3 install install ujson
RUN pip3 install --upgrade --no-cache-dir gdown
RUN pip3 install jupyterlab datasets 
RUN pip3 install rouge-score
RUN pip3 install evaluate
RUN pip3 install tqdm
RUN pip3 install huggingface_hub[hf_xet]

# TO COPY the data 
#COPY workspace /workspace
#Author Tosin