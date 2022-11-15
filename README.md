# Udacity DRL Project 1 - Navigation

### Introduction

For this project, an agent has been trained to navigate and collect bananas in a large, square world. The environment simulating the world was based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Noted that the environment was compiled based on toolkit version 0.4, provided by Udacity.

#### Environment introduction
1. State and action space: The state space is 37 including the agent's velocity, along with ray-based perception of objects around agent's forward direction, the action space is 4, ranging from 0 to 3. The discrete actions represent `forward, backward, left, right` for 0 to 3, respectively.

2. Reward is +1 for each yellow banana collected, and -1 for each blue ones collected.
3. Goal: the ultimate goal defined in my code is maximum undiscounted episodic reward larger than `23` and average reward over 10 episodes larger than `15` 
4. Bench implemented by Udacity: the bench agent training performance provided by Udacity indicates maximum socre circa 23, training episodes at circa 1700 eposides. 

### Getting Started adapted from Udacity instructions

1. Download the linux environment from the link below. 
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

2. Place the file in the DRLND GitHub repository, in the current folder where the codes are stored, and unzip (or decompress) the file. 
3. Install dependencies according the instructions
4. Setup the dependencies:
     ```bash
    git clone https://github.com/udacity/Value-based-methods.git
    cd Value-based-methods/python
    pip install .
    ```
5. Due to platform and cuda restrictions, the requirement from step 4 of torch==0.4.0 can no longer be satisfied in my machine. Instead I have losen the restriction to any torch versions that are suitable for current machine. The rest of the requirements remain unchanged and satisfied (including the unityagents version requirement). 

### Instructions
In `Navigation_training.ipynb`, the training process has been elaborated. The `agent.py` and `model.py` contains codes for RL agent and the backend neural net architecture, respectively. Check `Navigation_training.ipynb` for starter. 

