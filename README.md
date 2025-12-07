This folder contains scripts for running CARLA in headless mode.

## File structure
ROOT  
ROOT/Carla  
ROOT/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

## Entry script
run.sh (change ROOT directory here)

# Running MetaDrive Simulation
cd metadrive

pip install -e .

pip install "stable-baselines3[extra]"


python experiments/random_lane_keeping.py 0

python experiments/random_lane_keeping.py 1

python experiments/random_lane_keeping.py 2


python experiments/ppo_lane_keeping.py 0

python experiments/ppo_lane_keeping.py 1

python experiments/ppo_lane_keeping.py 2
