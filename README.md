## Deep Neural Networks from Scratch

An attempt to implement deep neural networks from scratch in C (and in the future other languages too), without relying on high-level machine learning frameworks.
This project is in constant WIP status as it is my sandbox for learning how to develop NNs from scratch.

## Overview

This repository contains an example implementation of a deep learning framework in C, featuring:

- **Modular layer architecture** - Linear (fully connected), ReLU, Leaky ReLU, and Sigmoid activation layers
- **Matrix operations library** - Vector-matrix multiplication and memory management
- **Backpropagation algorithm** - Gradient computation and weight updates
- **MNIST dataset support** - Binary file loader for handwritten digit classification
- **Model persistence** - Save and load trained models to/from checkpoint files

## Project Structure

```
C/
├── examples/
│   └── classifier.c          # MNIST digit classifier example
├── include/                   # Header files for all modules
├── src/
│   ├── layers/               # Neural network layer implementations
│   │   ├── linear/           # Fully connected layer
│   │   ├── relu/             # ReLU activation
│   │   ├── leakyrelu/        # Leaky ReLU activation
│   │   └── sigmoid/          # Sigmoid activation
│   └── lib/                  # Core libraries
│       ├── matrix.c          # Matrix/vector operations
│       ├── matmult.c         # Matrix multiplication
│       └── mnist.c           # MNIST dataset loader
└── tests/                    # Unit tests

```

## Features

### Neural Network Layers
- **Linear Layer**: Fully connected layer with configurable input/output dimensions and learning rate
- **ReLU**: Rectified Linear Unit activation (f(x) = max(0, x))
- **Leaky ReLU**: Leaky ReLU with configurable alpha parameter
- **Sigmoid**: Sigmoid activation for output normalization

## Building the Project

```bash
cd C
make clean
make
```

Executables will be created in the `bin/` directory.

## Running the Classifier

The main classifier trains a neural network on the MNIST dataset:

```bash
# Download MNIST dataset first (if not already present)
bash ../scripts/mnist_handwritten_download.sh

# Run the classifier
./bin/classifier
```

The default network architecture is:
- Input layer: 768 → 128 neurons + ReLU
- Hidden layer: 128 → 128 neurons + ReLU
- Output layer: 128 → 10 neurons + ReLU
- Training epochs: 5
- Learning rate: 4e-6

There is no current way of dynamically changing the network parameters. This is planned. If you need to test different scenarios, edit the `classifier.c` file and recompile.

## Test Cases

```bash
# Test matrix operations
./bin/matrix_test

# Test layer allocation/deallocation and checkpointing
./bin/allocate-deallocate

# Test forward and backward propagation
./bin/forward-backward

# Test MNIST data loading
./bin/mnist-load train-labels-idx1-ubyte train-images-idx3-ubyte
```

## Implementation Details

### Backpropagation
Gradients are computed using the chain rule and propagated backwards through the network:
- Compute loss gradient at output layer
- Propagate gradients through each layer
- Update weights using gradient descent: `w = w - learning_rate * gradient`
