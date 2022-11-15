# Udacity DRL Project 1 - Navigation

### Introduction

For this project, an agent has been trained to navigate and collect bananas in a large, square world. The environment simulating the world was based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Noted that the environment was compiled based on toolkit version 0.4, provided by Udacity.

#### Environment introduction
1. State and action space: The state space is 37 including the agent's velocity, along with ray-based perception of objects around agent's forward direction, the action space is 4, ranging from 0 to 3. The discrete actions represent `forward, backward, left, right` for 0 to 3, respectively.

2. Reward is +1 for each yellow banana collected, and -1 for each blue ones collected.
3. Success criteria: to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
4. Bench implemented by Udacity: the bench agent training performance provided by Udacity indicates maximum socre circa 23, training episodes at circa 1700 eposides. 

### Getting Started adapted from Udacity instructions

1. Copy this github repo:   
    ```
    git clone https://github.com/SDJoeKing/Value_based_navigation_reinforcement_learning.git
    ```
3. Download the linux environment from the link below. 
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

2. Place the file in the downloaded repo, and unzip (or decompress) the files. 
3. Activate the Python 36 virtual environment;
    ```
    At Linux terminal:
    source ~/your_python_36_env/bin/activation
    pip install numpy matplotlib scipy ipykernel pandas
    ```
5. While with python 36 activated, change to a different folder and setup the dependencies (noted gym environment is not required for this project):
     ```
    cd new_directory
    git clone https://github.com/udacity/Value-based-methods.git
    cd Value-based-methods/python
    pip install .
    ```
5. Due to platform and cuda restrictions, the requirement from step 5 of torch==0.4.0 can no longer be satisfied in my machine. Instead I have losen the restriction to any torch versions that are suitable for current machine. The rest of the requirements remain unchanged and satisfied (including the unityagents version requirement). 
6. Check `Navigation_training.ipynb`

### Instructions
In `Navigation_training.ipynb`, the training process has been elaborated. The `agent.py` and `model.py` contains codes for RL agent and the backend neural net architecture, respectively. Check `Navigation_training.ipynb` for starter. 

