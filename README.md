# Alphago Zero
This repository is based on https://github.com/tensorflow/minigo/tree/master. It extracts the python implementation of Alphago Zero. It provides a Dockerfile for setting up the environment, and a volume to be attached to the docker container for running the Alphago Zero.

## Paper Summary
AlphaGo Zero is trained through self-play reinforcement learning, starting from random play without any supervision or human data. Its design relies on a single neural network that combines both policy and value predictions, making the architecture more efficient.
