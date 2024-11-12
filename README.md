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
Eventually:
- Evaluate the Neural Network: Assess each new neural network checkpoint against the current best model to ensure high-quality data generation.

## Training implementation
We walk through the implementation of Alphago Zero here. An overview:
1. Bootstrap initial model: bootstrap.py initializes a model with random weights. Run this once, then run steps 2-3 in a loop until satisfied, then evaluate against previous models.
2. Self play: selfplay.py uses the latest model to play games against itself, generating data.
3. Training: train.py trains the model on all the cumulative games so far.
4. Evaluation: evaluate.py evaluates 2 different models
Note that we set default num_readouts as 100 and go.N as 9x9 for speed. Alter these in strategies.py and go.py as needed.

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

# Enter volume directory to execute following commands
cd volume
```

### Bootstrap
Initializes your working directory for the trainer and a random model. If these directories don't exist, bootstrap will create them for you.
```shell
python3 bootstrap.py \
  --work_dir /volume/work_dir \
  --export_path /volume/outputs/models/bootstrap \
  --create_bootstrap True
```

### Self play
This command starts self-playing, outputting its raw game data as tf.Examples as well as in SGF form in the directories.
```shell
python3 selfplay.py \
  --load_file=/volume/outputs/models/bootstrap \
  --selfplay_dir=/volume/outputs/data/selfplay \
  --holdout_dir=/volume/outputs/data/holdout \
  --sgf_dir=/volume/outputs/sgf
```

### Training
This command takes a directory of tf.Example files from selfplay and trains a new model, starting from the latest model weights in the estimator_working_dir parameter.
```shell
python3 train.py \
  /volume/outputs/data/selfplay/* \
  --work_dir=/volume/work_dir \
  --export_path=/volume/outputs/models/000001
```

### Evaluation
This command takes 
```shell
python3 evaluate.py \
  --black_model=/volume/outputs/models/000001 \
  --white_model=/volume/outputs/models/bootstrap \
  --eval_sgf_dir=/volume/outputs/eval_games \
  --num_evaluation_games=30 \
  --num_readouts=200
```

### Reinforcement learning loop
We have a script, rl_loop.py, that will perform the bootstrap initialization and training loop.
```shell
python3 rl_loop.py
```
After the loop, you can run evaluate.py against any earlier iteration to make sure the model has improved, as expected.
