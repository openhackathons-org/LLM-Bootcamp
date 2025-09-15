# NeMo-Challenge-Bootcamp

The NeMo Challenge Bootcamp is designed from a real-world perspective, following the data processing, development, and deployment pipeline paradigm. Attendees walk through the workflow of preprocessing a multi-turn conversational dataset for the summarization task and fine-tune the dataset on SOTA LLM using NeMo-Run. Attendees will also learn to optimize the fine-tuned model and apply prompt engineering techniques to solve complex real-world tasks. Furthermore, we introduced an AI Assistant customer care use case challenge to test attendees' understanding of the material and solidify their experience in the Text Generation domain.

## Deploying the Labs

### Prerequisites

To run this tutorial, you will need a Laptop/Workstation/DGX machine with a minimum of 40GB NVIDIA GPU.

- Install the latest [Docker](https://docs.docker.com/engine/install/) or [Singularity](https://sylabs.io/docs/).
- The labs require a Huggingface security token. Steps can be found [in the link here]( https://huggingface.co/docs/hub/en/security-tokens).


### Tested environment

We tested and ran all labs on a DGX machine equipped with an A100 and H100 GPUs (80GB).


### Deploying with container

You can deploy this material using Singularity and Docker containers. Please refer to the respective sections for the instructions.


#### Running Docker Container

To run the Labs, build a Docker container by following these steps:  

- Open a terminal window and navigate to the directory where `Dockerfile` file is located (e.g. `cd ~/NeMo-Challenge-Bootcamp`)
- To build the docker container, run : `sudo docker build -f Dockerfile --network=host -t <imagename>:<tagnumber> .`, for instance: 


```bash

sudo docker build -f Dockerfile --network=host -t nemodiv:v1 .

```
- To run the built container : 

```bash

docker run --rm -it --gpus all -p 8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
 -v ./nemo_workspace:/workspace nemodev:v1 
 jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```


flags:
- `--rm` will delete the container when finished.
- `-it` means run in interactive mode.
- `--gpus` option makes GPUs accessible inside the container.
- `-v` is used to mount host directories in the container filesystem.
- `--network=host` will share the host’s network stack to the container.
- `-p` flag explicitly maps a single port or range of ports.


Open the browser at `http://localhost:8888` and click on the `start_here.ipynb`. Go to the table of content and click on Lab 1: `Preprocessing Multi-turn Conversational Dataset`.
As soon as you are done with the rest of the labs, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl d` in the terminal window.



#### Running Singularity Container


- Build the Labs Singularity container with: 

```bash

singularity build --fakeroot --sandbox nemodev.simg Singularity


```

- To run the built container: 

```bash

singularity run --nv -B nemo_workspace:/workspace nemodev.simg 
jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```
 
 The `-B` flag mounts local directories in the container filesystem and ensures changes are stored locally in the project folder. Open jupyter lab in the browser: http://localhost:8888

You may start working on the labs by clicking the `start_here.ipynb` notebook.

When you finish these notebooks, shut down jupyter lab by selecting `File > Shut Down` in the top left corner, then shut down the Singularity container by typing `exit` or pressing `ctrl + d` in the terminal window.




## Known issues

- Your custom preprocessed dataset must be saved in `.jsonl` format and follow the naming convention as `training.jsonl`, `validation.jsonl`, and `test.jsonl` otherwise you will see an error stating: ` unable to find the training data ...`

- By default, NeMo stores the checkpoint here: `NEMO_MODELS_CACHE=/root/.cache/nemo/models`. You will be able to run inference successfully but run into errors runing infernece after restarting the NeMo container. This is because the cache as been cleared and an error will pop up stating: `Can't find the Llama-3.1-8B model...`

To prevent this, please set up the NeMo cache path to store the checkpoint in your local environment as: `os.environ["NEMO_MODELS_CACHE"] = "/workspace/model/"` 