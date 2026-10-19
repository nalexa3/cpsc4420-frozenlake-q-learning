# cpsc4420-frozenlake-q-learning
Repository containing the source code and documentation for CPSC 4420 Assignment 2.

## Author
Student: Nadia Alexander
Course: CPSC 4420  
Assignment: Artificial Intelligence Assignment 2


## Overview

This project implements a tabular Q-learning agent to solve the FrozenLake environment using Python and Gymnasium. The goal is to train an agent to navigate an 8×8 frozen lake grid from the starting state to the goal while avoiding holes.

## Environment
OpenAI Gymnasium: `FrozenLake-v1`  8x8 Map

## Requirements
- Python 3.10 or later
- Gymnasium
- NumPy
- Matplotlib
- imageio
- gymnasium[toy-text]

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/nalexa3/cpsc4420-frozenlake-q-learning.git
cd https://github.com/nalexa3/cpsc4420-frozenlake-q-learning.git
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

Activate the environment:
**macOS/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
python -m pip install gymnasium numpy matplotlib "gymnasium[toy-text]" imageio
```

## How to Run
```bash
python FrozenLake.py
```
