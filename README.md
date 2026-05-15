
<h1 align="center">
  Entmtp: accelerating LLM inference with entropy-aware multi-token-prediction 
</h1>

## Introduction
Our codebase is forked from the [Hydra](https://github.com/zankner/Hydra) codebase. If you find this work interesting you should check out their method!

Medusa introduces multiple lightweight draft heads on top of the frozen base LLM, which are used to predict multiple tokens ahead. This method reduces the size of speculative draft models, can utilize the high-quality representations of the base model, and is a simpler speculative framework. However, standard draft heads are only a function of the base LLM's hidden states from previously verified tokens, making them unaware of earlier tokens in the current candidate continuation.

Hydra improves upon Medusa by leveraging sequentially dependent draft heads that are aware of earlier tokens in the candidate continuation. This simple design change significantly improves the prediction quality of the heads, thus improving the overall decoding efficiency. We study these Hydra heads and alternate draft head architectures over a range of Vicuna models in the batch size 1 regime, achieving 2.5-2.7x improvements in throughput over baseline and 1.3x improvement in throughput over Medusa.

Entmtp builds on both lines of work and adds entropy-aware scheduling for Hydra decoding plus tooling to search, debias, and score custom greedy draft-tree topologies (beyond the default `mc_sim_7b_63` choice) for task-specific .

## Table of Contents
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Model Weights](#model-weights)
- [Inference](#inference)
- [Recalibrating Greedy Tree Topologies](#recalibrating-greedy-tree-topologies)
  - [Dataset](#dataset)
  - [Calibration Script](#calibration-script)
- [Evaluation](#evaluation)
- [Important Citations](#important-citations)
- [Important Files](#important-files)
- [Acknowledgements](#acknowledgements)

## Setup
```bash
git clone https://github.com/xikronz/entmtp
cd entmtp
pip install -e .
```

## Model Weights

| Base Model   | Hugging Face Repo                                                     |
| ----------  | --------------------------------------------------------------------- |
| Vicuna-7B   | [ankner/hydra-vicuna-7b-v1.3](https://huggingface.co/ankner/hydra-vicuna-7b-v1.3)   |
| Vicuna-13B  |  [ankner/hydra-vicuna-13b-v1.3](https://huggingface.co/ankner/hydra-vicuna-13b-v1.3) |
| Vicuna-33B  |  [ankner/hydra-vicuna-33b-v1.3](https://huggingface.co/ankner/hydra-vicuna-33b-v1.3) |

## Inference

The current inference script for Hydra supports inference at a batch size of 1, and we provide a demo CLI. We plan to support batched inference in the future.

The current cli command for running inference is

```bash
python -m hydra.inference.cli --model [HuggingFace repo / path of Hydra model]
```

Note that this script assumes the presence of one GPU, so you may have to set the ```CUDA_VISIBLE_DEVICES``` environment variable. 

## Recalibrating Greedy Tree Topologies 

First, install the training version of the repo.
```bash
pip install -e ".[train]"
```

### Dataset
Install `git-lfs` first:
```bash
apt-get install git-lfs
git lfs install
```

Then, install the ShareGPT dataset:
```bash
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```

Finally, create a train test split:
```bash
python hydra/data/partition_train_test.py
```

### Calibration Script

From the repository root (after `pip install -e .`), run greedy tree search on a calibration JSON/JSONL (ShareGPT split or other supported formats):

```bash
python entmtp/scripts/greedy_tree_search.py \
  --data-path /path/to/calibration_prompts.jsonl \
  --out-json /path/to/greedy_tree.json
```

For debiased acceptance on the Pareto frontier of an existing greedy run, see `entmtp/scripts/acceptance_frontier.py`.

## Evaluation

For evaluation results, please see the ```llm_judge/``` folder.

## Important Files
```hydra/model/hydra_model.py``` contains the ```HydraModel``` class which wraps all the decoding heads in this repository. We also have a variety of different heads, such as basic MLP and Attention-prefixed MLP layers in the ```hydra/model/hydra_heads/``` folder.

## Acknowledgements
This project is heavily influenced by the work done by [Medusa](https://github.com/FasterDecoding/Medusa/), [Hydra](https://github.com/zankner/Hydra), [Eagle](https://github.com/SafeAILab/EAGLE) and we would like to thank them for open-sourcing their codebase, which we have built off of.
