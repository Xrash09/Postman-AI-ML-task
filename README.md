# Postman-AI-ML-task
# NumPy Neural Network from Scratch - Postman Task 1

This repository contains a two-layer feedforward neural network built entirely from scratch using NumPy. It includes manual implementations of the forward pass, backpropagation, and a custom PyTorch testing harness to verify the mathematical accuracy of the gradients.

## 1. Repository Structure

* `neural_network.py`: The core network class containing the architecture, matrix math, and manual backpropagation algorithms.
* `train.py`: The data loading, preprocessing, training loop, and evaluation logic.
* `test_correctness.py`: An isolated testing harness that verifies the manual gradients against PyTorch's `autograd`.
* `WRITEUP.md`: A detailed breakdown of the mathematical derivations, architectural choices, and the debugging process.
* `loss_curve.png`: Generated automatically by `train.py` after a training run — a plot of cross-entropy loss over the 500 training iterations.

## 2. Prerequisites

To run this project, you will need Python 3 installed along with the following libraries:

```bash
pip install numpy pandas matplotlib torch
```

(Note: PyTorch is strictly used in `test_correctness.py` as a mathematical reference to verify the custom NumPy gradients).

## 3. Dataset Setup (Important)

This network is trained on the MNIST dataset in CSV format. To run the training script, you must download the dataset from Kaggle:

1. Visit the [Kaggle Digit Recognizer Data page](https://www.kaggle.com/c/digit-recognizer/data) (requires a Kaggle account).
2. Download the `train.csv` file.
3. Place the extracted `train.csv` file directly into the root folder of this repository.

## 4. Execution Instructions

### Training the Network

To train the model and see the loss and accuracy metrics print out every 10 iterations, run:

```bash
python train.py
```

This will train the network on 4,000 samples and evaluate it on 1,000 unseen test samples. It will also generate a `loss_curve.png` file in the root directory showing the training progress.

By default this runs for 500 iterations at a learning rate of 0.1. Both are just arguments passed to `gradient_descent()` inside `train.py` (`iterations=500, alpha=0.1`), so feel free to change them there if you want to experiment.

### Verifying the Gradients

To run the correctness harness and prove the manual backpropagation exactly matches PyTorch's `autograd` within a `1e-5` tolerance, run:

```bash
python test_correctness.py
```

This script isolates the forward and backward passes, feeds the same random seed to both NumPy and PyTorch, and compares the resulting `dW1`, `db1`, `dW2`, and `db2` matrices element-wise.

## 5. Reproducibility

Random seeds are fixed throughout — `np.random.seed(42)` in `train.py` and `torch.manual_seed(42)` in `test_correctness.py` — so re-running either script should reproduce the same results shown in `WRITEUP.md`.

## 6. Further Reading

Please see `WRITEUP.md` for a complete walkthrough of the cross-entropy and softmax calculus, as well as an honest look at the bugs encountered (and fixed) while scaling the gradients for mini-batch descent.
