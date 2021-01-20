##Flatland Challenge 2020 - PPO, CCPPO Implementation
Adapted from the Flatland Baseline
https://gitlab.aicrowd.com/flatland/neurips2020-flatland-baselines

### How to run the code:
Install the environment:

`conda env create -f environment-gpu.yml`

`conda activate flatland-baseline-gpu-env`

Run the experiment:

`python ./train.py -f experiment/ppo.yaml`

`python ./train.py -f experiment/ccppo_v1.yaml`