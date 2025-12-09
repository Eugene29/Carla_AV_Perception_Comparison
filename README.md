## Demo Video
https://youtu.be/rWp0JL7KcdY

------------------------------------------------------------------------

## Running MetaDrive Simulation

Follow the steps below to set up the MetaDrive environment, install
dependencies, run training, and visualize trained agents after cloning this repo

### 1. Navigate to the project directory

    cd metadrive-rl

### 2. Clone the MetaDrive repository

Clone the official MetaDrive repo into this folder:

    git clone https://github.com/metadriverse/metadrive.git

### 3. Install MetaDrive

Enter the MetaDrive directory and install it in editable mode:

    cd metadrive
    pip install -e .

### 4. Install Stable Baselines 3

Move back up one directory, then install SB3 with extras:

    cd ..
    pip install "stable-baselines3[extra]"

------------------------------------------------------------------------

## Training Agents

Use the following executables to train lane-keeping agents with
different algorithms and seeds.

### Random Policy Training

    python experiments/random_lane_keeping.py 0
    python experiments/random_lane_keeping.py 1
    python experiments/random_lane_keeping.py 2

### PPO Training

    python experiments/ppo_lane_keeping.py 0
    python experiments/ppo_lane_keeping.py 1
    python experiments/ppo_lane_keeping.py 2

### SAC Training

    python experiments/sac_lane_keeping.py 0
    python experiments/sac_lane_keeping.py 1
    python experiments/sac_lane_keeping.py 2

Trained agents are automatically saved in:

    results/

------------------------------------------------------------------------

## Visualizing a Trained Agent

Run the visualization script using the algorithm name and seed.

**Format:**

    python experiments/visualize_agent.py <algorithm> <seed>

**Example (PPO, seed 0):**

    python experiments/visualize_agent.py ppo 0
