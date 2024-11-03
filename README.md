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
- Optimize the Neural Network: Continuously train on data generated through self-play.
- Evaluator: Assess each new neural network checkpoint against the current best model to ensure high-quality data generation.

## Implementation
We walk through the implementation of Alphago Zero here.
- First
