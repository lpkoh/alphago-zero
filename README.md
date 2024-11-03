# Alphago Zero
This repository is based on https://github.com/tensorflow/minigo/tree/master. It extracts the python implementation of Alphago Zero. It provides a Dockerfile for setting up the environment, and a volume to be attached to the docker container for running the Alphago Zero.

## Paper Summary
AlphaGo Zero is trained through self-play reinforcement learning, starting from random play without any supervision or human data. Its design relies on a single neural network that combines both policy and value predictions, making the architecture more efficient.

### Key Features:
- Neural Network takes a board position s and outputs:
  - p: A vector of move probabilities, including the option to pass.
  - v: A scalar evaluation representing the probability of winning from the current board position.
- Uses a Monte Carlo tree search that uses this single neural network to evaluate position and sample moves.

### Reinforcement learning algorithm:
- In each position s, Monte Carlo Tree Search (MCTS) is executed, guided by the neural network, which outputs probabilities for each move. These search probabilities are typically much stronger than the raw move probabilities p directly provided by the neural network. As such, MCTS can be seen as a policy improvement operator.
- Self play with search, using improved MCTS based policy to select each move, then using the game winner z as a sample of the value, can be viewed as a policy evaluation operator.
- The primary concept of the reinforcement learning algorithm is to use these search operators in policy iteration, continuously updating the neural network parameters so that (p, v) closely matches the enhanced search probabilities and self-play outcomes.

### Self-Play Training Pipeline
Go through this loop:
- Self-Play: Generate training data by playing games using the latest model.
- Train the Neural Network: Train on data generated through self-play.
- Evaluate the Neural Network: Assess each new neural network checkpoint against the current best model to ensure high-quality data generation.

## Training implementation
We walk through the implementation of Alphago Zero here. An overview:
1. Bootstrap initial model: bootstrap.py initializes a model with random weights. Run this once, then run steps 2-4 in a loop until satisfied.
2. Self play: selfplay.py
3. Training: 
4. Evaluation:

### Setup
```shell
# Clone the AlphaGo Zero repository from GitHub
git clone https://github.com/lpkoh/alphago-zero.git

# Navigate into the cloned repository directory
cd alphago-zero

# Build the Docker image
# -t flag tags the image with the name 'alphago-zero'
docker build . -t alphago-zero

# Run the container interactively with volume mounting
docker run -it -v "$(pwd)/volume:/volume" alphago-zero /bin/bash
```

### Bootstrap
Initializes your working directory for the trainer and a random model. This random model is also exported to --model-save-path so that selfplay can immediately start playing with this random model. If these directories don't exist, bootstrap will create them for you.
```shell
python3 bootstrap.py --work_dir /volume/work_dir --export_path /volume/outputs/models/bootstrap --create_bootstrap True
```
